#![feature(iter_intersperse)]

mod compile;

macro_rules! c_str {
	($name:literal) => {
		#[allow(unused_unsafe)]
		unsafe {
			std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($name, "\0").as_bytes())
		}
	};
}
use std::{
	marker::PhantomData,
	sync::{
		atomic::{AtomicU64, Ordering},
		Arc,
		Mutex,
	},
	time::Duration,
};

use ash::vk::{self, Handle, TaggedStructure};
use c_str;
use notify_debouncer_full::{
	new_debouncer,
	notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher},
	DebounceEventResult,
	Debouncer,
	FileIdMap,
};
use rspirv::{
	binary::{Consumer, ParseAction, Parser},
	dr::{Instruction, ModuleHeader, Operand},
	spirv::{ExecutionModel, Op},
};

use crate::compile::ShaderBuilder;

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
	pub layout: vk::PipelineLayout,
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

pub struct GraphicsPipelineDescOwned {
	pub shaders: Vec<ShaderInfo>,
	pub raster: vk::PipelineRasterizationStateCreateInfo<'static>,
	pub depth: vk::PipelineDepthStencilStateCreateInfo<'static>,
	pub multisample: vk::PipelineMultisampleStateCreateInfo<'static>,
	pub blend: PipelineColorBlendStateCreateInfo,
	pub dynamic: Vec<vk::DynamicState>,
	pub layout: vk::PipelineLayout,
	pub color_attachments: Vec<vk::Format>,
	pub depth_attachment: vk::Format,
	pub stencil_attachment: vk::Format,
}

impl GraphicsPipelineDesc<'_> {
	pub fn owned(&self) -> GraphicsPipelineDescOwned {
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
			layout: self.layout,
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
			front_face: vk::FrontFace::CLOCKWISE,
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
			layout: vk::PipelineLayout::null(),
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

pub struct Pipeline(Arc<AtomicU64>, Arc<Mutex<RuntimeShared>>);

impl Pipeline {
	pub fn get(&self) -> vk::Pipeline { vk::Pipeline::from_raw(self.0.load(Ordering::Relaxed)) }

	pub unsafe fn destroy(self) {
		let s = self.1.lock().unwrap();
		(s.destroy_pipeline)(
			s.device,
			vk::Pipeline::from_raw(self.0.swap(0, Ordering::Relaxed)),
			std::ptr::null(),
		);
	}
}

enum PipelineDesc {
	Graphics(GraphicsPipelineDescOwned),
	Compute(vk::PipelineLayout, ShaderInfo),
}

struct RuntimeShared {
	builder: ShaderBuilder,
	pipelines: Vec<(PipelineDesc, Arc<AtomicU64>)>,
	recompiling: bool,
	device: vk::Device,
	create_graphics_pipelines: vk::PFN_vkCreateGraphicsPipelines,
	create_compute_pipelines: vk::PFN_vkCreateComputePipelines,
	destroy_pipeline: vk::PFN_vkDestroyPipeline,
}

impl RuntimeShared {
	fn get_shader(b: &mut ShaderBuilder, info: ShaderInfo) -> Result<(Vec<u32>, vk::ShaderStageFlags), String> {
		struct Dec(vk::ShaderStageFlags);
		impl Consumer for Dec {
			fn initialize(&mut self) -> ParseAction { ParseAction::Continue }

			fn finalize(&mut self) -> ParseAction { ParseAction::Continue }

			fn consume_header(&mut self, _: ModuleHeader) -> ParseAction { ParseAction::Continue }

			fn consume_instruction(&mut self, inst: Instruction) -> ParseAction {
				match inst.class.opcode {
					Op::EntryPoint => {
						for op in inst.operands {
							match op {
								Operand::ExecutionModel(m) => {
									self.0 = match m {
										ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
										ExecutionModel::TessellationControl => {
											vk::ShaderStageFlags::TESSELLATION_CONTROL
										},
										ExecutionModel::TessellationEvaluation => {
											vk::ShaderStageFlags::TESSELLATION_EVALUATION
										},
										ExecutionModel::Geometry => vk::ShaderStageFlags::GEOMETRY,
										ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
										ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
										ExecutionModel::Kernel => panic!("why do you have an opencl shader"),
										ExecutionModel::TaskNV => vk::ShaderStageFlags::TASK_NV,
										ExecutionModel::MeshNV => vk::ShaderStageFlags::MESH_NV,
										ExecutionModel::RayGenerationNV => vk::ShaderStageFlags::RAYGEN_NV,
										ExecutionModel::IntersectionNV => vk::ShaderStageFlags::INTERSECTION_NV,
										ExecutionModel::AnyHitNV => vk::ShaderStageFlags::ANY_HIT_NV,
										ExecutionModel::ClosestHitNV => vk::ShaderStageFlags::CLOSEST_HIT_NV,
										ExecutionModel::MissNV => vk::ShaderStageFlags::MISS_NV,
										ExecutionModel::CallableNV => vk::ShaderStageFlags::CALLABLE_NV,
										ExecutionModel::TaskEXT => vk::ShaderStageFlags::TASK_EXT,
										ExecutionModel::MeshEXT => vk::ShaderStageFlags::MESH_EXT,
									};
									break;
								},
								_ => {},
							}
						}
						ParseAction::Stop
					},
					_ => ParseAction::Continue,
				}
			}
		}

		let (module, entry) = info.shader.rsplit_once('.').unwrap();
		let spirv = b.load_module(module, entry, info.spec)?;
		let byte_slice = unsafe { std::slice::from_raw_parts(spirv.as_ptr() as _, spirv.len() * 4) };

		let mut dec = Dec(vk::ShaderStageFlags::empty());
		let _ = Parser::new(byte_slice, &mut dec).parse();
		// let mut iter = spirv.iter_mut();
		// while let Some(w) = iter.next() {
		// 	if *w == 14 | (3 << 16) {
		// 		break;
		// 	}
		// }
		// iter.next().unwrap();
		// *iter.next().unwrap() = 3;

		Ok((spirv, dec.0))
	}

	fn create_compute_pipeline(
		this: Arc<Mutex<Self>>, layout: vk::PipelineLayout, shader: ShaderInfo,
	) -> Result<Pipeline, vk::Result> {
		let mut t = this.lock().unwrap();
		let Self {
			ref mut builder,
			device,
			create_compute_pipelines,
			..
		} = *t;
		let p = match Self::create_compute_pipeline_inner(builder, device, create_compute_pipelines, layout, shader) {
			Ok(p) => p,
			Err(Ok(e)) => return Err(e),
			Err(Err(e)) => {
				println!("{e}");
				panic!("failed to compile shader");
			},
		};
		let inner = Arc::new(AtomicU64::new(p.as_raw()));
		t.pipelines.push((PipelineDesc::Compute(layout, shader), inner.clone()));
		drop(t);

		Ok(Pipeline(inner, this))
	}

	fn create_graphics_pipeline(this: Arc<Mutex<Self>>, desc: &GraphicsPipelineDesc) -> Result<Pipeline, vk::Result> {
		let mut t = this.lock().unwrap();
		let desc = desc.owned();
		let Self {
			ref mut builder,
			device,
			create_graphics_pipelines,
			..
		} = *t;
		let p = match Self::create_graphics_pipeline_inner(builder, device, create_graphics_pipelines, &desc) {
			Ok(p) => p,
			Err(Ok(e)) => return Err(e),
			Err(Err(e)) => {
				println!("{e}");
				panic!("failed to compile shader");
			},
		};
		let inner = Arc::new(AtomicU64::new(p.as_raw()));
		t.pipelines.push((PipelineDesc::Graphics(desc), inner.clone()));
		drop(t);

		Ok(Pipeline(inner, this))
	}

	fn create_compute_pipeline_inner(
		b: &mut ShaderBuilder, d: vk::Device, c: vk::PFN_vkCreateComputePipelines, layout: vk::PipelineLayout,
		shader: ShaderInfo,
	) -> Result<vk::Pipeline, Result<vk::Result, String>> {
		unsafe {
			let (code, stage) = Self::get_shader(b, shader).map_err(|x| Err(x))?;
			let mut pipeline = vk::Pipeline::null();
			let res = (c)(
				d,
				vk::PipelineCache::null(),
				1,
				&vk::ComputePipelineCreateInfo::default().layout(layout).stage(
					vk::PipelineShaderStageCreateInfo::default()
						.stage(stage)
						.name(c_str!("main"))
						.push_next(&mut vk::ShaderModuleCreateInfo::default().code(&code)),
				),
				std::ptr::null(),
				&mut pipeline,
			);

			match res {
				vk::Result::SUCCESS => Ok(pipeline),
				e => Err(Ok(e)),
			}
		}
	}

	fn create_graphics_pipeline_inner(
		b: &mut ShaderBuilder, d: vk::Device, c: vk::PFN_vkCreateGraphicsPipelines, desc: &GraphicsPipelineDescOwned,
	) -> Result<vk::Pipeline, Result<vk::Result, String>> {
		unsafe {
			let mut codes = Vec::with_capacity(desc.shaders.len());
			let mut infos = Vec::with_capacity(desc.shaders.len());
			let mut shaders = Vec::with_capacity(desc.shaders.len());
			for &s in desc.shaders.iter() {
				let (code, stage) = Self::get_shader(b, s).map_err(|x| Err(x))?;
				codes.push(code);
				shaders.push(
					vk::PipelineShaderStageCreateInfo::default()
						.stage(stage)
						.name(c_str!("main")),
				);
			}
			for code in codes.iter() {
				infos.push(vk::ShaderModuleCreateInfo::default().code(&code));
			}
			for (shader, info) in shaders.iter_mut().zip(infos.iter_mut()) {
				*shader = shader.push_next(info);
			}

			let mut pipeline = vk::Pipeline::null();
			let res = (c)(
				d,
				vk::PipelineCache::null(),
				1,
				&vk::GraphicsPipelineCreateInfo::default()
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
					.layout(desc.layout)
					.push_next(
						&mut vk::PipelineRenderingCreateInfo::default()
							.color_attachment_formats(&desc.color_attachments)
							.depth_attachment_format(desc.depth_attachment)
							.stencil_attachment_format(desc.stencil_attachment),
					),
				std::ptr::null(),
				&mut pipeline,
			);

			match res {
				vk::Result::SUCCESS => Ok(pipeline),
				e => Err(Ok(e)),
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
	// watcher: Debouncer<RecommendedWatcher, FileIdMap>,
	shared: Arc<Mutex<RuntimeShared>>,
}

impl ShaderRuntime {
	pub fn new<'s>(device: &ash::Device) -> Self {
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
		let shared = Arc::new(Mutex::new(RuntimeShared {
			builder: ShaderBuilder::new(source.clone(), cache.unwrap()).unwrap(),
			pipelines: Vec::new(),
			recompiling: false,
			device: device.handle(),
			create_graphics_pipelines: device.fp_v1_0().create_graphics_pipelines,
			create_compute_pipelines: device.fp_v1_0().create_compute_pipelines,
			destroy_pipeline: device.fp_v1_0().destroy_pipeline,
		}));
		let s = shared.clone();
		let mut watcher = new_debouncer(Duration::from_secs_f32(0.5), None, move |res: DebounceEventResult| {
			if let Ok(evs) = res {
				for ev in evs {
					if matches!(ev.kind, EventKind::Create(_) | EventKind::Modify(_)) {
						let mut s = s.lock().unwrap();
						s.recompiling = true;
						let RuntimeShared {
							ref mut builder,
							device,
							create_graphics_pipelines,
							create_compute_pipelines,
							destroy_pipeline,
							ref pipelines,
							..
						} = *s;
						for (desc, out) in pipelines.iter() {
							let new = match desc {
								PipelineDesc::Graphics(desc) => RuntimeShared::create_graphics_pipeline_inner(
									builder,
									device,
									create_graphics_pipelines,
									desc,
								),
								PipelineDesc::Compute(layout, shader) => RuntimeShared::create_compute_pipeline_inner(
									builder,
									device,
									create_compute_pipelines,
									*layout,
									*shader,
								),
							};
							match new {
								Ok(x) => {
									let old = out.swap(x.as_raw(), Ordering::Relaxed);
									unsafe {
										(destroy_pipeline)(device, vk::Pipeline::from_raw(old), std::ptr::null());
									}
								},
								Err(Ok(e)) => println!("failed to recompile pipeline: {e:?}"),
								Err(Err(e)) => println!("{e}"),
							}
						}
						s.recompiling = false;
					}
				}
				println!();
			}
		})
		.unwrap();
		// let _ = watcher.watcher().watch(&source, RecursiveMode::Recursive);

		Self {
			// watcher,
			shared,
		}
	}

	pub fn create_graphics_pipeline(&self, desc: GraphicsPipelineDesc) -> Result<Pipeline, vk::Result> {
		RuntimeShared::create_graphics_pipeline(self.shared.clone(), &desc)
	}

	pub fn create_compute_pipeline(
		&self, layout: vk::PipelineLayout, shader: ShaderInfo,
	) -> Result<Pipeline, vk::Result> {
		RuntimeShared::create_compute_pipeline(self.shared.clone(), layout, shader)
	}

	pub fn status(&self) -> HotreloadStatus {
		match self.shared.lock().unwrap().recompiling {
			true => HotreloadStatus::Recompiling,
			false => HotreloadStatus::Waiting,
		}
	}
}
