use std::io::Write;

use ash::vk;
use bytemuck::{bytes_of, NoUninit, PodInOption, ZeroableInOption};
use radiance_graph::{
	device::{
		descriptor::{SamplerId, StorageImageId},
		Device,
	},
	graph::{
		BufferDesc,
		BufferUsage,
		BufferUsageType,
		ExternalImage,
		Frame,
		ImageDesc,
		ImageUsage,
		ImageUsageType,
		PassBuilder,
		PassContext,
		Res,
	},
	resource::{self, BufferHandle, Image, ImageView, Resource, Subresource},
	sync::Shader,
};
use vek::Vec2;

use crate::{
	asset::{rref::RRef, scene::Scene},
	mesh::{Camera, CameraData, RenderInfo},
};

struct Persistent {
	scene: RRef<Scene>,
	camera: Camera,
	hzb: Image,
}

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

pub struct Setup {
	inner: Option<Persistent>,
}

impl Setup {
	pub fn new() -> Self { Self { inner: None } }

	pub fn run(&mut self, frame: &mut Frame, info: &RenderInfo, hzb_sampler: SamplerId) -> Resources {
		let needs_clear = self.init_persistent(frame, info);
		let Persistent {
			hzb, camera: prev_cam, ..
		} = self.inner.as_mut().unwrap();
		let res = info.size;
		let cam = info.camera;
		let prev = *prev_cam;
		*prev_cam = cam;

		let mut pass = frame.pass("setup cull buffers");
		let camera = pass.resource(
			BufferDesc {
				size: std::mem::size_of::<CameraData>() as u64 * 2,
				upload: true,
			},
			BufferUsage { usages: &[] },
		);
		let hzb = pass.resource(
			ExternalImage {
				handle: hzb.handle(),
				layout: if needs_clear {
					vk::ImageLayout::UNDEFINED
				} else {
					vk::ImageLayout::GENERAL
				},
				desc: hzb.desc(),
			},
			ImageUsage {
				format: vk::Format::UNDEFINED,
				usages: if needs_clear {
					&[ImageUsageType::TransferWrite]
				} else {
					&[ImageUsageType::General]
				},
				view_type: None,
				subresource: Subresource::default(),
			},
		);
		let late_instances = pass.resource(
			BufferDesc {
				size: ((info.scene.instance_count() as usize + 4) * std::mem::size_of::<u32>()) as _,
				upload: false,
			},
			BufferUsage {
				usages: &[BufferUsageType::TransferWrite],
			},
		);
		let size = ((12 * 1024 * 1024 + 2) * 2 * std::mem::size_of::<u32>()) as _;
		let bvh_queues = [(); 3].map(|_| {
			pass.resource(
				BufferDesc { size, upload: false },
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			)
		});
		let meshlet_queues = [(); 2].map(|_| {
			pass.resource(
				BufferDesc { size, upload: false },
				BufferUsage {
					usages: &[BufferUsageType::TransferWrite],
				},
			)
		});
		let meshlet_render_lists = [(); 4].map(|_| {
			pass.resource(
				BufferDesc { size, upload: false },
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

	fn init_persistent(&mut self, frame: &mut Frame, info: &RenderInfo) -> bool {
		let size = info.size / 2;
		match &mut self.inner {
			Some(Persistent { scene, hzb, .. }) => {
				let curr_size = hzb.desc().size;
				if curr_size.width != size.x || curr_size.height != size.y {
					frame.delete(self.inner.take().unwrap().hzb);
					self.inner = Some(Persistent {
						scene: info.scene.clone(),
						camera: info.camera,
						hzb: Self::make_hzb(frame.device(), size),
					});
					true
				} else if !scene.ptr_eq(&info.scene) {
					*scene = info.scene.clone();
					true
				} else {
					false
				}
			},
			None => {
				self.inner = Some(Persistent {
					scene: info.scene.clone(),
					camera: info.camera,
					hzb: Self::make_hzb(frame.device(), size),
				});
				true
			},
		}
	}

	fn make_hzb(device: &Device, size: Vec2<u32>) -> Image {
		Image::create(
			device,
			resource::ImageDesc {
				name: "persistent hzb",
				flags: vk::ImageCreateFlags::empty(),
				format: vk::Format::R32_SFLOAT,
				size: vk::Extent3D {
					width: size.x,
					height: size.y,
					depth: 1,
				},
				levels: size.x.max(size.y).ilog2(),
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
				usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
			},
		)
		.unwrap()
	}

	pub unsafe fn destroy(self, device: &Device) {
		if let Some(p) = self.inner {
			p.hzb.destroy(device);
		}
	}
}
