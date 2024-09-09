use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, NoUninit, PodInOption, ZeroableInOption};
use radiance_asset::rref::RWeak;
use radiance_graph::{
	device::descriptor::{SamplerId, StorageImageId},
	graph::{
		BufferDesc,
		BufferLoc,
		BufferUsage,
		BufferUsageType,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Res,
	},
	resource::{BufferHandle, ImageView, Subresource},
	sync::Shader,
};

use crate::mesh::{Camera, CameraData, RenderInfo};

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
	pub camera: Res<BufferHandle>,
	pub hzb: Res<ImageView>,
	pub hzb_sampler: SamplerId,
	pub late_instances: Res<BufferHandle>,
	pub len: u32,
	pub bvh_queues: [Res<BufferHandle>; 3],
	pub meshlet_queues: [Res<BufferHandle>; 2],
	pub meshlet_render_lists: [Res<BufferHandle>; 4],
	pub visbuffer: Res<ImageView>,
	pub debug: Option<DebugRes>,
}

impl Resources {
	pub fn camera(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(
			self.camera,
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Compute)],
			},
		);
		self.camera
	}

	pub fn camera_mesh(&self, pass: &mut PassBuilder) -> Res<BufferHandle> {
		pass.reference(
			self.camera,
			BufferUsage {
				usages: &[BufferUsageType::ShaderStorageRead(Shader::Mesh)],
			},
		);
		self.camera
	}

	pub fn hzb(&self, pass: &mut PassBuilder) -> Res<ImageView> {
		pass.reference(
			self.hzb,
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::ShaderReadSampledImage(Shader::Compute)],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
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

	pub fn mesh(&self, pass: &mut PassBuilder) -> [Res<BufferHandle>; 4] {
		for m in self.meshlet_render_lists {
			pass.reference(
				m,
				BufferUsage {
					usages: &[
						BufferUsageType::IndirectBuffer,
						BufferUsageType::ShaderStorageRead(Shader::Mesh),
						BufferUsageType::ShaderStorageRead(Shader::Compute),
					],
				},
			);
		}
		self.meshlet_render_lists
	}

	pub fn output(&self, pass: &mut PassBuilder, buf: Res<BufferHandle>) -> Res<BufferHandle> {
		pass.reference(
			buf,
			BufferUsage {
				usages: &[
					BufferUsageType::ShaderStorageRead(Shader::Compute),
					BufferUsageType::ShaderStorageWrite(Shader::Compute),
				],
			},
		);
		buf
	}

	pub fn visbuffer(&self, pass: &mut PassBuilder) -> Res<ImageView> {
		pass.reference(
			self.visbuffer,
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: &[ImageUsageType::ShaderStorageWrite(Shader::Fragment)],
				view_type: Some(vk::ImageViewType::TYPE_2D),
				subresource: Subresource::default(),
			},
		);
		self.visbuffer
	}

	pub fn debug(&self, pass: &mut PassBuilder) -> Option<DebugRes> {
		if let Some(d) = self.debug {
			pass.reference(
				d.overdraw,
				ImageUsage {
					format: vk::Format::UNDEFINED,
					usages: &[ImageUsageType::ShaderStorageWrite(Shader::Fragment)],
					view_type: Some(vk::ImageViewType::TYPE_2D),
					subresource: Subresource::default(),
				},
			);
			pass.reference(
				d.hwsw,
				ImageUsage {
					format: vk::Format::UNDEFINED,
					usages: &[ImageUsageType::ShaderStorageWrite(Shader::Fragment)],
					view_type: Some(vk::ImageViewType::TYPE_2D),
					subresource: Subresource::default(),
				},
			);
		}
		self.debug
	}
}

struct Persistent {
	scene: RWeak,
	camera: Camera,
}

pub struct Setup {
	inner: Option<Persistent>,
}

impl Setup {
	pub fn new() -> Self { Self { inner: None } }

	pub fn run(&mut self, frame: &mut Frame, info: &RenderInfo, hzb_sampler: SamplerId) -> Resources {
		let (needs_clear, prev) = match &mut self.inner {
			Some(Persistent { scene, camera }) => {
				let prev = *camera;
				*camera = info.camera;
				let sc = info.scene.downgrade();
				if *scene != sc {
					*scene = sc;
					(true, prev)
				} else {
					(false, prev)
				}
			},
			None => {
				self.inner = Some(Persistent {
					scene: info.scene.downgrade(),
					camera: info.camera,
				});
				(true, info.camera)
			},
		};
		let res = info.size;
		let cam = info.camera;

		let mut pass = frame.pass("setup cull buffers");
		let camera = pass.resource(
			BufferDesc {
				size: std::mem::size_of::<CameraData>() as u64 * 2,
				loc: BufferLoc::Upload,
				persist: None,
			},
			BufferUsage { usages: &[] },
		);
		let size = info.size / 2;
		let hzb = pass.resource(
			ImageDesc {
				format: vk::Format::R32_SFLOAT,
				size: vk::Extent3D {
					width: size.x,
					height: size.y,
					depth: 1,
				},
				levels: size.x.max(size.y).ilog2(),
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				persist: Some("persistent hzb"),
			},
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: if needs_clear {
					&[ImageUsageType::TransferWrite]
				} else {
					&[]
				},
				view_type: None,
				subresource: Subresource::default(),
			},
		);
		let late_instances = pass.resource(
			BufferDesc {
				size: ((info.scene.instance_count() as usize + 4) * std::mem::size_of::<u32>()) as _,
				loc: BufferLoc::GpuOnly,
				persist: None,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		let size = ((12 * 1024 * 1024 + 2) * 2 * std::mem::size_of::<u32>()) as _;
		let bvh_queues = [(); 3].map(|_| {
			pass.resource(
				BufferDesc {
					size,
					loc: BufferLoc::GpuOnly,
					persist: None,
				},
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			)
		});
		let meshlet_queues = [(); 2].map(|_| {
			pass.resource(
				BufferDesc {
					size,
					loc: BufferLoc::GpuOnly,
					persist: None,
				},
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			)
		});
		let meshlet_render_lists = [(); 4].map(|_| {
			pass.resource(
				BufferDesc {
					size,
					loc: BufferLoc::GpuOnly,
					persist: None,
				},
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			)
		});
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
		let usage = ImageUsage {
			format: vk::Format::UNDEFINED,
			usages: &[ImageUsageType::TransferWrite],
			view_type: None,
			subresource: Subresource::default(),
		};
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

		pass.build(move |mut pass| unsafe {
			let dev = pass.device.device();
			let buf = pass.buf;
			let mut writer = pass.get(camera).data.as_mut();
			let aspect = res.x as f32 / res.y as f32;
			let cd = CameraData::new(aspect, cam);
			let prev_cd = CameraData::new(aspect, prev);
			writer.write(bytes_of(&cd)).unwrap();
			writer.write(bytes_of(&prev_cd)).unwrap();

			if needs_clear {
				dev.cmd_clear_color_image(
					buf,
					pass.get(hzb).image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&vk::ClearColorValue::default(),
					&[vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.base_mip_level(0)
						.level_count(vk::REMAINING_MIP_LEVELS)
						.base_array_layer(0)
						.layer_count(vk::REMAINING_ARRAY_LAYERS)],
				);
			}
			dev.cmd_clear_color_image(
				buf,
				pass.get(visbuffer).image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&vk::ClearColorValue {
					uint32: [u32::MAX, 0, 0, 0],
				},
				&[vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_mip_level(0)
					.level_count(vk::REMAINING_MIP_LEVELS)
					.base_array_layer(0)
					.layer_count(vk::REMAINING_ARRAY_LAYERS)],
			);
			if let Some(d) = debug {
				dev.cmd_clear_color_image(
					buf,
					pass.get(d.overdraw).image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&vk::ClearColorValue::default(),
					&[vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.base_mip_level(0)
						.level_count(vk::REMAINING_MIP_LEVELS)
						.base_array_layer(0)
						.layer_count(vk::REMAINING_ARRAY_LAYERS)],
				);
				dev.cmd_clear_color_image(
					buf,
					pass.get(d.hwsw).image,
					vk::ImageLayout::TRANSFER_DST_OPTIMAL,
					&vk::ClearColorValue::default(),
					&[vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.base_mip_level(0)
						.level_count(vk::REMAINING_MIP_LEVELS)
						.base_array_layer(0)
						.layer_count(vk::REMAINING_ARRAY_LAYERS)],
				);
			}

			for b in bvh_queues.into_iter().chain(meshlet_queues).chain(Some(late_instances)) {
				dev.cmd_update_buffer(buf, pass.get(b).buffer, 0, bytes_of(&[0u32, 0, 1, 1]));
			}

			for b in meshlet_render_lists {
				dev.cmd_update_buffer(buf, pass.get(b).buffer, 0, bytes_of(&[0u32, 1, 1]));
			}
		});

		Resources {
			camera,
			hzb,
			hzb_sampler,
			late_instances,
			len: size as _,
			bvh_queues,
			meshlet_queues,
			meshlet_render_lists,
			visbuffer,
			debug,
		}
	}
}
