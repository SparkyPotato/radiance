use std::{
	panic,
	sync::Arc,
	thread::{self, JoinHandle},
};

use ash::vk;
use crossbeam_channel::{Receiver, Sender};
use half::f16;
use radiance_asset_runtime::{rref::RRef, scene::Scene};
use radiance_core::{CoreDevice, CoreFrame};
use radiance_graph::{
	device::QueueType,
	graph::{ImageDesc, ImageUsage, ImageUsageType, Res},
	resource::ImageView,
};
use radiance_util::staging::ImageStage;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use tracing::{span, Level};
use vek::{
	num_traits::{One, Zero},
	Mat4,
	Ray,
	Rgba,
	Vec2,
	Vec3,
	Vec4,
};

use crate::{cpu_path::framebuffer::Framebuffer, mesh::visbuffer::Camera};

mod framebuffer;

#[derive(Clone)]
pub struct RenderInfo {
	pub scene: RRef<Scene>,
	pub camera: Camera,
	pub size: Vec2<u32>,
}

pub struct CpuPath {
	cmds: Sender<Command>,
	director: Option<JoinHandle<()>>,
	curr_camera: Camera,
	curr_framebuffer: Arc<Framebuffer>,
	curr_scene: Option<RRef<Scene>>,
}

#[derive(Clone)]
enum Command {
	Render,
	SceneChange(RRef<Scene>),
	CameraChange(Camera),
	FramebufferChange(Arc<Framebuffer>),
	Quit,
}

impl CpuPath {
	pub fn new() -> Self {
		let curr_framebuffer = Arc::new(Framebuffer::new(Vec2::zero()));
		let (cmds, rx) = crossbeam_channel::unbounded();
		let c = curr_framebuffer.clone();
		let director = thread::spawn(|| director(rx, c));

		Self {
			cmds,
			director: Some(director),
			curr_camera: Camera::default(),
			curr_framebuffer,
			curr_scene: None,
		}
	}

	pub fn run<'pass>(
		&'pass mut self, device: &'pass CoreDevice, frame: &mut CoreFrame<'pass, '_>, info: RenderInfo,
	) -> Res<ImageView> {
		if self.curr_camera != info.camera {
			self.curr_camera = info.camera;
			self.cmds.send(Command::CameraChange(info.camera)).unwrap();
		}
		if self.curr_framebuffer.size() != info.size {
			self.curr_framebuffer = Arc::new(Framebuffer::new(info.size));
			self.cmds
				.send(Command::FramebufferChange(self.curr_framebuffer.clone()))
				.unwrap();
		}
		if let Some(s) = &self.curr_scene {
			if !s.ptr_eq(&info.scene) {
				self.curr_scene = Some(info.scene.clone());
				self.cmds.send(Command::SceneChange(info.scene)).unwrap();
			}
		} else {
			self.curr_scene = Some(info.scene.clone());
			self.cmds.send(Command::SceneChange(info.scene)).unwrap();
		}
		self.cmds.send(Command::Render).unwrap();

		let mut pass = frame.pass("CPU Path");
		let size = self.curr_framebuffer.size();
		let size = vk::Extent3D {
			width: size.x,
			height: size.y,
			depth: 1,
		};
		let out = pass.output(
			ImageDesc {
				size,
				levels: 1,
				layers: 1,
				samples: vk::SampleCountFlags::TYPE_1,
			},
			ImageUsage {
				format: vk::Format::R16G16B16A16_SFLOAT,
				usages: &[ImageUsageType::TransferWrite],
				view_type: vk::ImageViewType::TYPE_2D,
				aspect: vk::ImageAspectFlags::COLOR,
			},
		);
		pass.build(move |mut ctx| {
			let image = ctx.get(out).image;
			ctx.ctx
				.stage(device, |sctx, _| {
					sctx.stage_image(
						&self.curr_framebuffer.read.lock().unwrap(),
						image,
						ImageStage {
							buffer_row_length: 0,
							buffer_image_height: 0,
							image_subresource: vk::ImageSubresourceLayers {
								aspect_mask: vk::ImageAspectFlags::COLOR,
								mip_level: 0,
								base_array_layer: 0,
								layer_count: 1,
							},
							image_offset: vk::Offset3D::default(),
							image_extent: size,
						},
						true,
						QueueType::Graphics,
						// TODO: Maybe remove the general usage but it doesn't matter because we're
						// hammering the CPU, not the GPU here.
						&[ImageUsageType::General],
						&[ImageUsageType::General],
					)
				})
				.unwrap();
		});
		out
	}
}

impl Drop for CpuPath {
	fn drop(&mut self) {
		self.cmds.send(Command::Quit).unwrap();
		if let Err(e) = self.director.take().unwrap().join() {
			panic::resume_unwind(e);
		}
	}
}

fn director(rx: Receiver<Command>, mut framebuffer: Arc<Framebuffer>) {
	let mut camera = Camera::default();
	let mut view = Mat4::default();
	let mut scene = None;
	let mut renders_left = 10;

	loop {
		if renders_left == 0 {
			match rx.recv().unwrap() {
				Command::Render => renders_left = 5,
				Command::SceneChange(s) => scene = Some(s),
				Command::FramebufferChange(f) => framebuffer = f,
				Command::CameraChange(c) => {
					camera = c;
					view = c.view.inverted();
				},
				Command::Quit => return,
			}
		}
		while let Ok(t) = rx.try_recv() {
			match t {
				Command::Render => renders_left = 5,
				Command::SceneChange(s) => scene = Some(s),
				Command::FramebufferChange(f) => framebuffer = f,
				Command::CameraChange(c) => {
					camera = c;
					view = c.view.inverted();
				},
				Command::Quit => return,
			}
		}

		if let Some(ref scene) = scene {
			let s = framebuffer.size().map(|x| x as f32);
			let proj = Mat4::perspective_fov_lh_zo(camera.fov, s.x, s.y, camera.near, 10.0).inverted();
			framebuffer.get_tiles().for_each_init(
				|| SmallRng::from_entropy(),
				|rng, tile| {
					let _s = span!(Level::TRACE, "trace tile");
					let _e = _s.enter();

					for p in tile {
						let pixel = p.pixel.map(|x| x as f32) + Vec2::new(rng.gen(), rng.gen());
						let clip = pixel / framebuffer.size().map(|x| x as f32) * 2.0 - 1.0;
						let target = (proj * Vec4::new(clip.x, -clip.y, 1.0, 1.0)).xyz().normalized();
						let ray = Ray {
							origin: (view * Vec3::zero().with_w(1.0)).xyz(),
							direction: (view * target.with_w(0.0)).xyz(),
						};

						*p.data = if scene.intersect(ray, f32::INFINITY).is_some() {
							Rgba::one()
						} else {
							Rgba::new(f16::zero(), f16::zero(), f16::zero(), f16::one())
						};
					}
				},
			);
			framebuffer.present();
		}
		renders_left -= 1;
	}
}
