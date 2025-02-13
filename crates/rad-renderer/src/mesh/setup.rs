use ash::vk;
use bytemuck::{offset_of, NoUninit, PodInOption, ZeroableInOption};
use rad_graph::{
	device::descriptor::{SamplerId, StorageImageId},
	graph::{
		BufferDesc,
		BufferUsage,
		BufferUsageType,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Persist,
		Res,
	},
	resource::{Buffer, BufferHandle, Image, ImageView, Subresource},
	sync::Shader,
};
use tracing::error;
use vek::Vec2;

use crate::{
	mesh::{CullStats, RenderInfo},
	scene::{camera::CameraScene, virtual_scene::VirtualScene, WorldRenderer},
};

#[derive(Copy, Clone)]
pub struct DebugRes {
	pub overdraw: Res<ImageView>,
	pub hwsw: Res<ImageView>,
}

impl DebugRes {
	pub fn get(self, pass: &mut PassContext) -> DebugResId {
		DebugResId {
			overdraw: pass.get(self.overdraw).storage_id.unwrap(),
			hwsw: pass.get(self.hwsw).storage_id.unwrap(),
		}
	}
}

#[derive(Copy, Clone, NoUninit)]
#[repr(C)]
pub struct DebugResId {
	pub overdraw: StorageImageId,
	pub hwsw: StorageImageId,
}

unsafe impl PodInOption for DebugResId {}
unsafe impl ZeroableInOption for DebugResId {}

pub struct Resources {
	pub scene: VirtualScene,
	pub camera: Res<BufferHandle>,
	pub hzb: Res<ImageView>,
	pub hzb_sampler: SamplerId,
	pub late_instances: Res<BufferHandle>,
	pub bvh_queues: [Res<BufferHandle>; 2],
	pub meshlet_queue: Res<BufferHandle>,
	pub meshlet_render: Res<BufferHandle>,
	pub stats: Res<BufferHandle>,
	pub visbuffer: Res<ImageView>,
	pub debug: Option<DebugRes>,
	pub res: Vec2<u32>,
}

impl Resources {
	pub fn instances(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(self.scene.instances, BufferUsage::read(Shader::Compute));
		self.scene.instances
	}

	pub fn instances_mesh(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(self.scene.instances, BufferUsage::read(Shader::Mesh));
		self.scene.instances
	}

	pub fn camera(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(self.camera, BufferUsage::read(Shader::Compute));
		self.camera
	}

	pub fn camera_mesh(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(self.camera, BufferUsage::read(Shader::Mesh));
		self.camera
	}

	pub fn hzb(&self, pass: &mut PassBuilder) -> Res<ImageView> {
		pass.reference(self.hzb, ImageUsage::sampled_2d(Shader::Compute));
		self.hzb
	}

	pub fn input(&self, pass: &mut PassBuilder, buf: Res<BufferHandle>) -> Res<BufferHandle> {
		pass.reference(
			buf,
			BufferUsage {
				usages: &[
					BufferUsageType::IndirectBuffer,
					BufferUsageType::ShaderStorageRead(Shader::Compute),
				],
			},
		);
		buf
	}

	pub fn output(&self, pass: &mut PassBuilder, buf: Res<BufferHandle>) -> Res<BufferHandle> {
		pass.reference(buf, BufferUsage::read_write(Shader::Compute));
		buf
	}

	pub fn input_output(&self, pass: &mut PassBuilder, buf: Res<BufferHandle>) -> Res<BufferHandle> {
		pass.reference(
			buf,
			BufferUsage {
				usages: &[
					BufferUsageType::IndirectBuffer,
					BufferUsageType::ShaderStorageRead(Shader::Compute),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);
		buf
	}

	pub fn mesh(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(
			self.meshlet_render,
			BufferUsage {
				usages: &[
					BufferUsageType::IndirectBuffer,
					BufferUsageType::ShaderStorageRead(Shader::Mesh),
					BufferUsageType::ShaderStorageRead(Shader::Compute),
				],
			},
		);
		self.meshlet_render
	}

	pub fn mesh_zero(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(self.meshlet_render, BufferUsage::transfer_write());
		self.meshlet_render
	}

	pub fn stats(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(self.stats, BufferUsage::write(Shader::Compute));
		self.stats
	}

	pub fn stats_mesh(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(
			self.stats,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageWrite(Shader::Mesh),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);
		self.stats
	}

	pub fn visbuffer(&self, pass: &mut PassBuilder) -> Res<ImageView> {
		pass.reference(self.visbuffer, ImageUsage::write_2d(Shader::Fragment));
		self.visbuffer
	}

	pub fn debug(&self, pass: &mut PassBuilder) -> Option<DebugRes> {
		if let Some(d) = self.debug {
			pass.reference(d.overdraw, ImageUsage::write_2d(Shader::Fragment));
			pass.reference(d.hwsw, ImageUsage::write_2d(Shader::Fragment));
		}
		self.debug
	}
}

pub struct Setup {
	pub stats: CullStats,
	hzb: Persist<Image>,
	stats_readback: Persist<Buffer>,
}

fn prev_pot(x: u32) -> u32 { 1 << x.ilog2() }

impl Setup {
	pub fn new() -> Self {
		Self {
			stats: CullStats::default(),
			hzb: Persist::new(),
			stats_readback: Persist::new(),
		}
	}

	pub fn run<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, rend: &mut WorldRenderer<'pass, '_>, info: &RenderInfo,
		hzb_sampler: SamplerId,
	) -> Resources {
		let scene = rend.get::<VirtualScene>(frame);
		let camera = rend.get::<CameraScene>(frame);

		// TODO: handle world change.
		let needs_clear = camera.prev.camera != camera.curr.camera;
		let res = info.size;

		let mut pass = frame.pass("setup cull buffers");
		let size = info.size.map(prev_pot);
		let hzb = pass.resource(
			ImageDesc {
				size: vk::Extent3D {
					width: size.x,
					height: size.y,
					depth: 1,
				},
				format: vk::Format::R32_SFLOAT,
				levels: size.x.max(size.y).ilog2(),
				persist: Some(self.hzb),
				..Default::default()
			},
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: if needs_clear {
					&[ImageUsageType::TransferWrite]
				} else {
					&[
						ImageUsageType::AddUsage(vk::ImageUsageFlags::TRANSFER_DST),
						ImageUsageType::OverrideLayout(vk::ImageLayout::READ_ONLY_OPTIMAL),
					]
				},
				view_type: None,
				subresource: Subresource::default(),
			},
		);

		let usage = BufferUsage::transfer_write();
		let late_instances = pass.resource(
			BufferDesc::gpu(((scene.instance_count as usize + 4) * std::mem::size_of::<u32>()) as _),
			usage,
		);
		let bvh_count = 2 * 1024 * 1024u32;
		let meshlet_count = 32 * 1024 * 1024u32;
		let render_count = 16 * 1024 * 1024u32;
		let desc = |count| BufferDesc::gpu((count as u64 * 2 + 9) * std::mem::size_of::<u32>() as u64);
		let bvh_queues = [(); 2].map(|_| pass.resource(desc(bvh_count), usage));
		let meshlet_queue = pass.resource(desc(meshlet_count), usage);
		let meshlet_render = pass.resource(desc(render_count), usage);
		let stats = pass.resource(
			BufferDesc::readback(std::mem::size_of::<CullStats>() as u64, self.stats_readback),
			BufferUsage::transfer_write(),
		);

		let desc = ImageDesc {
			size: vk::Extent3D {
				width: res.x,
				height: res.y,
				depth: 1,
			},
			format: vk::Format::R64_UINT,
			levels: 1,
			layers: 1,
			samples: vk::SampleCountFlags::TYPE_1,
			persist: None,
		};
		let usage = ImageUsage::transfer_write();
		let visbuffer = pass.resource(desc, usage);
		let debug = info.debug_info.then(|| {
			let overdraw = pass.resource(
				ImageDesc {
					format: vk::Format::R32_UINT,
					..desc
				},
				usage,
			);
			let hwsw = pass.resource(
				ImageDesc {
					format: vk::Format::R64_UINT,
					..desc
				},
				usage,
			);
			DebugRes { overdraw, hwsw }
		});

		pass.build(move |mut pass| {
			if needs_clear | pass.is_uninit(hzb) {
				pass.zero(hzb);
			}
			pass.clear_image(
				visbuffer,
				vk::ClearColorValue {
					uint32: [u32::MAX, 0, 0, 0],
				},
			);
			if let Some(d) = debug {
				pass.zero(d.overdraw);
				pass.zero(d.hwsw);
			}

			for (b, count) in bvh_queues
				.into_iter()
				.map(|x| (x, bvh_count))
				.chain([(meshlet_queue, meshlet_count), (meshlet_render, render_count)])
			{
				pass.update_buffer(b, 0, &[count, 0, 0, 1, 1, 0, 0, 1, 1]);
			}
			pass.update_buffer(late_instances, 0, &[0u32, 0, 1, 1]);

			self.stats = pass.readback(stats, 0);
			if self.stats.overflow != 0 {
				error!("Cull queues overflowed");
			}
			pass.fill_buffer(stats, 0, offset_of!(self.stats, CullStats, overflow) as _, 4);
		});

		Resources {
			scene,
			camera: camera.buf,
			hzb,
			hzb_sampler,
			late_instances,
			bvh_queues,
			meshlet_queue,
			meshlet_render,
			stats,
			visbuffer,
			debug,
			res: info.size,
		}
	}
}
