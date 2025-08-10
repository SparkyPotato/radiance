use std::{
	marker::PhantomData,
	sync::{
		Arc,
		Mutex,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	time::Duration,
};

use ash::vk::{self, Handle, TaggedStructure};
use notify_debouncer_full::{
	DebounceEventResult,
	Debouncer,
	FileIdMap,
	new_debouncer,
	notify::{EventKind, RecommendedWatcher, RecursiveMode},
};

use crate::{
	Error,
	device::{Device, shader::compile::ShaderBuilder},
	resource::{Buffer, BufferDesc, BufferType, Resource},
};

mod compile;

#[derive(Copy, Clone, Default)]
pub struct ShaderInfo {
	pub shader: &'static str,
	pub spec: &'static [&'static str],
}

pub struct GraphicsPipelineDesc<'a> {
	pub shaders: &'a [ShaderInfo],
	pub raster: vk::PipelineRasterizationStateCreateInfo<'static>,
	pub depth: vk::PipelineDepthStencilStateCreateInfo<'static>,
	pub multisample: vk::PipelineMultisampleStateCreateInfo<'static>,
	pub blend: vk::PipelineColorBlendStateCreateInfo<'a>,
	pub dynamic: &'a [vk::DynamicState],
	pub color_attachments: &'a [vk::Format],
	pub depth_attachment: vk::Format,
	pub stencil_attachment: vk::Format,
}

pub struct PipelineColorBlendStateCreateInfo {
	pub flags: vk::PipelineColorBlendStateCreateFlags,
	pub logic_op_enable: vk::Bool32,
	pub logic_op: vk::LogicOp,
	pub attachments: Vec<vk::PipelineColorBlendAttachmentState>,
	pub blend_constants: [f32; 4],
}

struct GraphicsPipelineDescOwned {
	shaders: Vec<ShaderInfo>,
	raster: vk::PipelineRasterizationStateCreateInfo<'static>,
	depth: vk::PipelineDepthStencilStateCreateInfo<'static>,
	multisample: vk::PipelineMultisampleStateCreateInfo<'static>,
	blend: PipelineColorBlendStateCreateInfo,
	dynamic: Vec<vk::DynamicState>,
	color_attachments: Vec<vk::Format>,
	depth_attachment: vk::Format,
	stencil_attachment: vk::Format,
}

impl GraphicsPipelineDesc<'_> {
	fn owned(&self) -> GraphicsPipelineDescOwned {
		GraphicsPipelineDescOwned {
			shaders: self.shaders.to_vec(),
			raster: self.raster,
			depth: self.depth,
			multisample: self.multisample,
			blend: PipelineColorBlendStateCreateInfo {
				flags: self.blend.flags,
				logic_op_enable: self.blend.logic_op_enable,
				logic_op: self.blend.logic_op,
				attachments: unsafe {
					std::slice::from_raw_parts(self.blend.p_attachments, self.blend.attachment_count as _)
				}
				.to_vec(),
				blend_constants: self.blend.blend_constants,
			},
			dynamic: self.dynamic.to_vec(),
			color_attachments: self.color_attachments.to_vec(),
			depth_attachment: self.depth_attachment,
			stencil_attachment: self.stencil_attachment,
		}
	}
}

impl Default for GraphicsPipelineDesc<'_> {
	fn default() -> Self {
		const BLEND: vk::PipelineColorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo {
			s_type: vk::PipelineColorBlendStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineColorBlendStateCreateFlags::empty(),
			logic_op_enable: 0,
			logic_op: vk::LogicOp::NO_OP,
			attachment_count: 0,
			p_attachments: std::mem::align_of::<vk::PipelineColorBlendAttachmentState>() as _,
			blend_constants: [0.0, 0.0, 0.0, 0.0],
			_marker: PhantomData,
		};

		const RASTER: vk::PipelineRasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo {
			s_type: vk::PipelineRasterizationStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineRasterizationStateCreateFlags::empty(),
			depth_clamp_enable: 0,
			rasterizer_discard_enable: 0,
			polygon_mode: vk::PolygonMode::FILL,
			cull_mode: vk::CullModeFlags::BACK,
			front_face: vk::FrontFace::COUNTER_CLOCKWISE,
			depth_bias_enable: 0,
			depth_bias_constant_factor: 0.0,
			depth_bias_clamp: 0.0,
			depth_bias_slope_factor: 0.0,
			line_width: 1.0,
			_marker: PhantomData,
		};

		const STENCIL: vk::StencilOpState = vk::StencilOpState {
			fail_op: vk::StencilOp::REPLACE,
			pass_op: vk::StencilOp::REPLACE,
			depth_fail_op: vk::StencilOp::REPLACE,
			compare_op: vk::CompareOp::NEVER,
			compare_mask: 0,
			write_mask: 0,
			reference: 0,
		};

		const DEPTH: vk::PipelineDepthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo {
			s_type: vk::PipelineDepthStencilStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
			depth_test_enable: 0,
			depth_write_enable: 0,
			depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
			depth_bounds_test_enable: 0,
			stencil_test_enable: 0,
			front: STENCIL,
			back: STENCIL,
			min_depth_bounds: 0.0,
			max_depth_bounds: 1.0,
			_marker: PhantomData,
		};

		const MULTISAMPLE: vk::PipelineMultisampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo {
			s_type: vk::PipelineMultisampleStateCreateInfo::STRUCTURE_TYPE,
			p_next: std::ptr::null(),
			flags: vk::PipelineMultisampleStateCreateFlags::empty(),
			rasterization_samples: vk::SampleCountFlags::TYPE_1,
			sample_shading_enable: 0,
			min_sample_shading: 0.0,
			p_sample_mask: std::ptr::null(),
			alpha_to_coverage_enable: 0,
			alpha_to_one_enable: 0,
			_marker: PhantomData,
		};

		Self {
			shaders: &[],
			color_attachments: &[],
			blend: BLEND,
			// Values that can be defaulted below.
			depth_attachment: vk::Format::UNDEFINED,
			stencil_attachment: vk::Format::UNDEFINED,
			raster: RASTER,
			depth: DEPTH,
			multisample: MULTISAMPLE,
			dynamic: &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
		}
	}
}

#[derive(Copy, Clone)]
pub enum RtShaderGroup {
	General(u32),
	Triangles {
		closest_hit: Option<u32>,
		any_hit: Option<u32>,
	},
	Procedural {
		intersection: u32,
		any_hit: Option<u32>,
		closest_hit: Option<u32>,
	},
}

#[derive(Copy, Clone, Default)]
pub struct RtPipelineDesc<'a> {
	pub shaders: &'a [ShaderInfo],
	pub groups: &'a [RtShaderGroup],
	pub recursion_depth: u32,
}

impl RtPipelineDesc<'_> {
	fn owned(&self) -> RtPipelineDescOwned {
		RtPipelineDescOwned {
			shaders: self.shaders.to_vec(),
			groups: self.groups.to_vec(),
			recursion_depth: self.recursion_depth,
		}
	}
}

struct RtPipelineDescOwned {
	shaders: Vec<ShaderInfo>,
	groups: Vec<RtShaderGroup>,
	recursion_depth: u32,
}

pub struct GraphicsPipeline(Arc<AtomicU64>, Device);
pub struct ComputePipeline(Arc<AtomicU64>, Device);
struct RtPipelineData {
	pipeline: vk::Pipeline,
	sbt: Buffer,
	rgen: vk::StridedDeviceAddressRegionKHR,
	hit: vk::StridedDeviceAddressRegionKHR,
	miss: vk::StridedDeviceAddressRegionKHR,
	callable: vk::StridedDeviceAddressRegionKHR,
}
pub struct RtPipeline(Arc<Mutex<RtPipelineData>>, Device);

impl GraphicsPipeline {
	pub fn bind(&self, buf: vk::CommandBuffer) {
		unsafe {
			self.1.device().cmd_bind_pipeline(
				buf,
				vk::PipelineBindPoint::GRAPHICS,
				vk::Pipeline::from_raw(self.0.load(Ordering::Relaxed)),
			);
		}
	}

	pub unsafe fn destroy(self) {
		unsafe {
			self.1
				.device()
				.destroy_pipeline(vk::Pipeline::from_raw(self.0.swap(0, Ordering::Relaxed)), None);
		}
	}
}

impl ComputePipeline {
	pub fn bind(&self, buf: vk::CommandBuffer) {
		unsafe {
			self.1.device().cmd_bind_pipeline(
				buf,
				vk::PipelineBindPoint::COMPUTE,
				vk::Pipeline::from_raw(self.0.load(Ordering::Relaxed)),
			);
		}
	}

	pub unsafe fn destroy(self) {
		unsafe {
			self.1
				.device()
				.destroy_pipeline(vk::Pipeline::from_raw(self.0.swap(0, Ordering::Relaxed)), None);
		}
	}
}

impl RtPipeline {
	pub fn bind(&self, buf: vk::CommandBuffer) {
		unsafe {
			self.1.device().cmd_bind_pipeline(
				buf,
				vk::PipelineBindPoint::RAY_TRACING_KHR,
				self.0.lock().unwrap().pipeline,
			);
		}
	}

	pub fn trace_rays(&self, buf: vk::CommandBuffer, width: u32, height: u32, depth: u32) {
		unsafe {
			let m = self.0.lock().unwrap();
			self.1
				.rt_ext()
				.cmd_trace_rays(buf, &m.rgen, &m.miss, &m.hit, &m.callable, width, height, depth);
		}
	}

	pub fn trace_rays_indirect(&self, buf: vk::CommandBuffer, addr: vk::DeviceAddress) {
		unsafe {
			let m = self.0.lock().unwrap();
			self.1
				.rt_ext()
				.cmd_trace_rays_indirect(buf, &m.rgen, &m.miss, &m.hit, &m.callable, addr);
		}
	}

	pub unsafe fn destroy(self) {
		unsafe {
			let mut m = self.0.lock().unwrap();
			self.1.device().destroy_pipeline(m.pipeline, None);
			m.pipeline = vk::Pipeline::null();
			std::mem::take(&mut m.sbt).destroy(&self.1);
		}
	}
}

enum PipelineData {
	Graphics(GraphicsPipelineDescOwned, Arc<AtomicU64>),
	Compute(ShaderInfo, bool, Arc<AtomicU64>),
	Rt(RtPipelineDescOwned, Arc<Mutex<RtPipelineData>>),
}

struct PipelineCompiler {
	device: Device,
	builder: ShaderBuilder,
}

impl PipelineCompiler {
	fn get_shader(&mut self, info: ShaderInfo) -> Result<(Vec<u32>, vk::ShaderStageFlags), String> {
		let (module, entry) = info.shader.rsplit_once('.').unwrap();
		self.builder.load_module(module, entry, info.spec)
	}

	#[track_caller]
	fn compile_graphics(&mut self, desc: &GraphicsPipelineDescOwned) -> Result<vk::Pipeline, Result<Error, String>> {
		unsafe {
			let mut codes = Vec::with_capacity(desc.shaders.len());
			let mut infos = Vec::with_capacity(desc.shaders.len());
			let mut shaders = Vec::with_capacity(desc.shaders.len());
			for &s in desc.shaders.iter() {
				let (code, stage) = self.get_shader(s).map_err(Err)?;
				codes.push(code);
				shaders.push(vk::PipelineShaderStageCreateInfo::default().stage(stage).name(c"main"));
			}
			for code in codes.iter() {
				infos.push(vk::ShaderModuleCreateInfo::default().code(code));
			}
			for (shader, info) in shaders.iter_mut().zip(infos.iter_mut()) {
				*shader = shader.push_next(info);
			}

			self.device
				.device()
				.create_graphics_pipelines(
					vk::PipelineCache::null(),
					&[vk::GraphicsPipelineCreateInfo::default()
						.stages(&shaders)
						.vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
						.input_assembly_state(
							&vk::PipelineInputAssemblyStateCreateInfo::default()
								.topology(vk::PrimitiveTopology::TRIANGLE_LIST),
						)
						.viewport_state(
							&vk::PipelineViewportStateCreateInfo::default()
								.viewports(&[vk::Viewport::default()])
								.scissors(&[vk::Rect2D::default()]),
						)
						.rasterization_state(&desc.raster)
						.depth_stencil_state(&desc.depth)
						.multisample_state(&desc.multisample)
						.color_blend_state(
							&vk::PipelineColorBlendStateCreateInfo::default()
								.flags(desc.blend.flags)
								.logic_op_enable(desc.blend.logic_op_enable != 0)
								.logic_op(desc.blend.logic_op)
								.attachments(&desc.blend.attachments)
								.blend_constants(desc.blend.blend_constants),
						)
						.dynamic_state(&vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&desc.dynamic))
						.layout(self.device.layout())
						.push_next(
							&mut vk::PipelineRenderingCreateInfo::default()
								.color_attachment_formats(&desc.color_attachments)
								.depth_attachment_format(desc.depth_attachment)
								.stencil_attachment_format(desc.stencil_attachment),
						)],
					None,
				)
				.map(|x| x[0])
				.map_err(|(_, e)| Ok(e.into()))
		}
	}

	#[track_caller]
	fn compile_compute(
		&mut self, shader: ShaderInfo, force_wave_32: bool,
	) -> Result<vk::Pipeline, Result<Error, String>> {
		unsafe {
			let (code, stage) = self.get_shader(shader).map_err(Err)?;
			let mut info = vk::ShaderModuleCreateInfo::default().code(&code);
			let mut stage = vk::PipelineShaderStageCreateInfo::default()
				.stage(stage)
				.name(c"main")
				.push_next(&mut info);
			let mut req_size =
				vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
			if force_wave_32 {
				stage = stage.push_next(&mut req_size);
			}
			self.device
				.device()
				.create_compute_pipelines(
					vk::PipelineCache::null(),
					&[vk::ComputePipelineCreateInfo::default()
						.layout(self.device.layout())
						.stage(stage)],
					None,
				)
				.map(|x| x[0])
				.map_err(|(_, e)| Ok(e.into()))
		}
	}

	#[track_caller]
	fn compile_rt(&mut self, desc: &RtPipelineDescOwned) -> Result<RtPipelineData, Result<Error, String>> {
		unsafe {
			let mut codes = Vec::with_capacity(desc.shaders.len());
			let mut infos = Vec::with_capacity(desc.shaders.len());
			let mut shaders = Vec::with_capacity(desc.shaders.len());
			for &s in desc.shaders.iter() {
				let (code, stage) = self.get_shader(s).map_err(Err)?;
				codes.push(code);
				shaders.push(vk::PipelineShaderStageCreateInfo::default().stage(stage).name(c"main"));
			}
			for code in codes.iter() {
				infos.push(vk::ShaderModuleCreateInfo::default().code(code));
			}
			for (shader, info) in shaders.iter_mut().zip(infos.iter_mut()) {
				*shader = shader.push_next(info);
			}

			let groups = desc
				.groups
				.iter()
				.map(|&g| match g {
					RtShaderGroup::General(i) => vk::RayTracingShaderGroupCreateInfoKHR::default()
						.ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
						.general_shader(i)
						.closest_hit_shader(vk::SHADER_UNUSED_KHR)
						.any_hit_shader(vk::SHADER_UNUSED_KHR)
						.intersection_shader(vk::SHADER_UNUSED_KHR),
					RtShaderGroup::Triangles { closest_hit, any_hit } => {
						vk::RayTracingShaderGroupCreateInfoKHR::default()
							.ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
							.general_shader(vk::SHADER_UNUSED_KHR)
							.closest_hit_shader(closest_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
							.any_hit_shader(any_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
							.intersection_shader(vk::SHADER_UNUSED_KHR)
					},
					RtShaderGroup::Procedural {
						intersection,
						any_hit,
						closest_hit,
					} => vk::RayTracingShaderGroupCreateInfoKHR::default()
						.ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
						.general_shader(vk::SHADER_UNUSED_KHR)
						.closest_hit_shader(closest_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
						.any_hit_shader(any_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
						.intersection_shader(intersection),
				})
				.collect::<Vec<_>>();

			let pipeline = self
				.device
				.rt_ext()
				.create_ray_tracing_pipelines(
					vk::DeferredOperationKHR::null(),
					vk::PipelineCache::null(),
					&[vk::RayTracingPipelineCreateInfoKHR::default()
						.flags(vk::PipelineCreateFlags::empty())
						.stages(&shaders)
						.groups(&groups)
						.max_pipeline_ray_recursion_depth(desc.recursion_depth)
						.layout(self.device.layout())],
					None,
				)
				.map_err(|(_, e)| Ok(e.into()))?[0];

			let mut props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
			let mut p = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
			self.device
				.instance()
				.get_physical_device_properties2(self.device.physical_device(), &mut p);
			let handle_size = props.shader_group_handle_size as u64;
			let handle_align = props.shader_group_handle_alignment as u64;
			let base_align = props.shader_group_base_alignment as u64;
			let handles = self
				.device
				.rt_ext()
				.get_ray_tracing_shader_group_handles(
					pipeline,
					0,
					groups.len() as _,
					handle_size as usize * groups.len(),
				)
				.map_err(|e| Ok(e.into()))?;

			let rgen_groups = desc.groups.iter().enumerate().filter_map(|(i, &g)| {
				if let RtShaderGroup::General(r) = g
					&& shaders[r as usize].stage == vk::ShaderStageFlags::RAYGEN_KHR
				{
					Some(i)
				} else {
					None
				}
			});
			let hit_groups = desc.groups.iter().enumerate().filter_map(|(i, &g)| {
				matches!(g, RtShaderGroup::Triangles { .. } | RtShaderGroup::Procedural { .. }).then_some(i)
			});
			let miss_groups = desc.groups.iter().enumerate().filter_map(|(i, &g)| {
				if let RtShaderGroup::General(r) = g
					&& shaders[r as usize].stage == vk::ShaderStageFlags::MISS_KHR
				{
					Some(i)
				} else {
					None
				}
			});
			let callable_groups = desc.groups.iter().enumerate().filter_map(|(i, &g)| {
				if let RtShaderGroup::General(r) = g
					&& shaders[r as usize].stage == vk::ShaderStageFlags::CALLABLE_KHR
				{
					Some(i)
				} else {
					None
				}
			});

			let rgen_count = rgen_groups.clone().count() as u64;
			let hit_count = hit_groups.clone().count() as u64;
			let miss_count = miss_groups.clone().count() as u64;
			let callable_count = callable_groups.clone().count() as u64;
			assert_eq!(rgen_count, 1, "only one rgen group is supported");

			let handle_size_aligned = align_up(handle_size, handle_align);
			let st = align_up(handle_size_aligned, base_align);
			let mut rgen = vk::StridedDeviceAddressRegionKHR::default().stride(st).size(st);
			let mut hit = vk::StridedDeviceAddressRegionKHR::default()
				.stride(handle_size_aligned)
				.size(align_up(hit_count * handle_size_aligned, base_align));
			let mut miss = vk::StridedDeviceAddressRegionKHR::default()
				.stride(handle_size_aligned)
				.size(align_up(miss_count * handle_size_aligned, base_align));
			let mut callable = vk::StridedDeviceAddressRegionKHR::default()
				.stride(handle_size_aligned)
				.size(align_up(callable_count * handle_size_aligned, base_align));

			let sbt = Buffer::create(
				&self.device,
				BufferDesc {
					name: "rt sbt",
					size: rgen.size + hit.size + miss.size + callable.size,
					ty: BufferType::Gpu,
				},
			)
			.map_err(|e| Ok(e))?;
			rgen.device_address = sbt.ptr::<()>().addr();
			hit.device_address = rgen.device_address + rgen.size;
			miss.device_address = hit.device_address + hit.size;
			callable.device_address = miss.device_address + miss.size;

			let b = sbt.data().as_ptr() as *mut u8;
			let mut p = b;
			for r in rgen_groups {
				std::ptr::copy_nonoverlapping(handles.as_ptr().add(r * handle_size as usize), p, handle_size as usize);
				p = p.add(handle_size_aligned as usize);
			}
			p = b.add((hit.device_address - rgen.device_address) as _);
			for r in hit_groups {
				std::ptr::copy_nonoverlapping(handles.as_ptr().add(r * handle_size as usize), p, handle_size as usize);
				p = p.add(handle_size_aligned as usize);
			}
			p = b.add((miss.device_address - rgen.device_address) as _);
			for r in miss_groups {
				std::ptr::copy_nonoverlapping(handles.as_ptr().add(r * handle_size as usize), p, handle_size as usize);
				p = p.add(handle_size_aligned as usize);
			}
			p = b.add((callable.device_address - rgen.device_address) as _);
			for r in callable_groups {
				std::ptr::copy_nonoverlapping(handles.as_ptr().add(r * handle_size as usize), p, handle_size as usize);
				p = p.add(handle_size_aligned as usize);
			}

			Ok(RtPipelineData {
				pipeline,
				sbt,
				rgen,
				hit,
				miss,
				callable,
			})
		}
	}
}

fn align_up(size: u64, align: u64) -> u64 { (size + align - 1) & !(align - 1) }

struct RuntimeShared {
	pipelines: Vec<PipelineData>,
	compiler: PipelineCompiler,
}

impl RuntimeShared {
	#[track_caller]
	fn create_graphics_pipeline(&mut self, desc: GraphicsPipelineDesc) -> crate::Result<GraphicsPipeline> {
		let desc = desc.owned();
		let p = match self.compiler.compile_graphics(&desc) {
			Ok(p) => p,
			Err(Ok(e)) => return Err(e),
			Err(Err(e)) => return Err(e.into()),
		};
		let inner = Arc::new(AtomicU64::new(p.as_raw()));
		self.pipelines.push(PipelineData::Graphics(desc, inner.clone()));

		Ok(GraphicsPipeline(inner, self.compiler.device.clone()))
	}

	#[track_caller]
	fn create_compute_pipeline(&mut self, shader: ShaderInfo, force_wave_32: bool) -> crate::Result<ComputePipeline> {
		let p = match self.compiler.compile_compute(shader, force_wave_32) {
			Ok(p) => p,
			Err(Ok(e)) => return Err(e),
			Err(Err(e)) => return Err(e.into()),
		};
		let inner = Arc::new(AtomicU64::new(p.as_raw()));
		self.pipelines
			.push(PipelineData::Compute(shader, force_wave_32, inner.clone()));

		Ok(ComputePipeline(inner, self.compiler.device.clone()))
	}

	#[track_caller]
	fn create_rt_pipeline(&mut self, desc: RtPipelineDesc) -> crate::Result<RtPipeline> {
		let desc = desc.owned();
		let data = match self.compiler.compile_rt(&desc) {
			Ok(d) => d,
			Err(Ok(e)) => return Err(e),
			Err(Err(e)) => return Err(e.into()),
		};
		let inner = Arc::new(Mutex::new(data));
		self.pipelines.push(PipelineData::Rt(desc, inner.clone()));

		Ok(RtPipeline(inner, self.compiler.device.clone()))
	}

	fn recompile_pipelines(&mut self) {
		let Self { pipelines, compiler } = self;
		if let Err(e) = compiler.builder.reload() {
			println!("failed to recompile pipeline: {e:?}");
		}
		for data in pipelines.iter() {
			// TODO: the GPU is still using the old pipeline, so we can't destroy it
			// yet
			let err = match data {
				PipelineData::Graphics(desc, out) => compiler.compile_graphics(desc).map(|x| {
					out.swap(x.as_raw(), Ordering::Relaxed);
				}),
				PipelineData::Compute(shader, force_wave_32, out) => {
					compiler.compile_compute(*shader, *force_wave_32).map(|x| {
						out.swap(x.as_raw(), Ordering::Relaxed);
					})
				},
				PipelineData::Rt(desc, out) => compiler.compile_rt(desc).map(|data| {
					*out.lock().unwrap() = data;
				}),
			};
			match err {
				Ok(_) => {},
				Err(Ok(e)) => println!("failed to recompile pipeline: {e:?}"),
				Err(Err(e)) => println!("{e}"),
			}
		}
	}
}

pub enum HotreloadStatus {
	Waiting,
	Recompiling,
	Errored,
}

pub struct ShaderRuntime {
	_watcher: Debouncer<RecommendedWatcher, FileIdMap>,
	status: Arc<AtomicBool>,
	shared: Arc<Mutex<RuntimeShared>>,
}

impl ShaderRuntime {
	pub fn new<'s>(device: Device) -> Self {
		let c = std::env::current_exe().unwrap();
		let mut curr = c.as_path();
		let mut source = None;
		let mut cache = None;
		while let Some(p) = curr.parent() {
			if p.join("shaders").exists() && p.join("target").exists() {
				source = Some(p.join("shaders"));
				let c = p.join("target/shaders");
				std::fs::create_dir_all(&c).unwrap();
				cache = Some(c);
				break;
			}
			curr = p;
		}
		let source = source.unwrap();
		let status = Arc::new(AtomicBool::new(false));
		let shared = Arc::new(Mutex::new(RuntimeShared {
			pipelines: Vec::new(),
			compiler: PipelineCompiler {
				device,
				builder: ShaderBuilder::new(source.clone(), cache.unwrap()).unwrap(),
			},
		}));
		let s = shared.clone();
		let st = status.clone();
		let mut watcher = new_debouncer(Duration::from_secs_f32(0.5), None, move |res: DebounceEventResult| {
			if let Ok(evs) = res
				&& evs
					.into_iter()
					.any(|ev| matches!(ev.kind, EventKind::Create(_) | EventKind::Modify(_)))
			{
				st.store(true, Ordering::Relaxed);
				s.lock().unwrap().recompile_pipelines();
				st.store(false, Ordering::Relaxed);
				println!();
			}
		})
		.unwrap();
		let _ = watcher.watch(&source, RecursiveMode::Recursive);

		Self {
			_watcher: watcher,
			status,
			shared,
		}
	}

	#[track_caller]
	pub fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc) -> crate::Result<GraphicsPipeline> {
		self.shared.lock().unwrap().create_graphics_pipeline(desc)
	}

	#[track_caller]
	pub fn create_compute_pipeline(&self, shader: ShaderInfo, force_wave_32: bool) -> crate::Result<ComputePipeline> {
		self.shared
			.lock()
			.unwrap()
			.create_compute_pipeline(shader, force_wave_32)
	}

	#[track_caller]
	pub fn create_rt_pipeline(&self, desc: RtPipelineDesc) -> crate::Result<RtPipeline> {
		self.shared.lock().unwrap().create_rt_pipeline(desc)
	}

	pub fn status(&self) -> HotreloadStatus {
		match self.status.load(Ordering::Relaxed) {
			true => HotreloadStatus::Recompiling,
			false => HotreloadStatus::Waiting,
		}
	}
}
