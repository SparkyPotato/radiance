use std::marker::PhantomData;

use ash::vk;
use bytemuck::NoUninit;

use crate::{
	device::{Device, GraphicsPipeline, GraphicsPipelineDesc, ShaderInfo},
	graph::{PassContext, Res},
	resource::ImageView,
	util::{
		pass::{Attachment, Load},
		pipeline::{no_blend, no_cull, simple_blend},
	},
	Result,
};

pub struct RenderPass<T> {
	pipeline: GraphicsPipeline,
	y_up: bool,
	_phantom: PhantomData<fn() -> T>,
}

impl<T: NoUninit> RenderPass<T> {
	pub fn new(device: &Device, desc: GraphicsPipelineDesc, y_up: bool) -> Result<Self> {
		Ok(Self {
			pipeline: device.graphics_pipeline(desc)?,
			y_up,
			_phantom: PhantomData,
		})
	}

	fn setup(&self, pass: &mut crate::util::pass::RenderPass, push: &T) {
		pass.bind_graphics(&self.pipeline);
		pass.push(0, push);
	}

	pub fn start<'a, 'frame, 'graph>(
		&self, pass: &'a mut PassContext<'frame, 'graph>, push: &T, attachments: &[Attachment],
		depth: Option<&Attachment>,
	) -> crate::util::pass::RenderPass<'a, 'frame, 'graph> {
		let mut pass = pass.render_pass(self.y_up, attachments, depth);
		self.setup(&mut pass, push);
		pass
	}

	pub fn start_empty<'a, 'frame, 'graph>(
		&self, pass: &'a mut PassContext<'frame, 'graph>, push: &T, size: vk::Extent2D,
	) -> crate::util::pass::RenderPass<'a, 'frame, 'graph> {
		let mut pass = pass.empty_render_pass(self.y_up, size);
		self.setup(&mut pass, push);
		pass
	}

	pub unsafe fn destroy(self) { self.pipeline.destroy(); }
}

pub struct FullscreenPass<T> {
	inner: RenderPass<T>,
	_phantom: PhantomData<fn() -> T>,
}

impl<T: NoUninit> FullscreenPass<T> {
	pub fn new(device: &Device, pixel: ShaderInfo, attachments: &[vk::Format]) -> Result<Self> {
		let blends: Vec<_> = attachments.iter().map(|_| no_blend()).collect();
		Ok(Self {
			inner: RenderPass::new(
				device,
				GraphicsPipelineDesc {
					shaders: &[
						ShaderInfo {
							shader: "graph.util.screen",
							..Default::default()
						},
						pixel,
					],
					raster: no_cull(),
					blend: simple_blend(&blends),
					color_attachments: attachments,
					..Default::default()
				},
				false,
			)?,
			_phantom: PhantomData,
		})
	}

	pub fn run_one<'a, 'frame, 'graph>(
		&self, pass: &'a mut PassContext<'frame, 'graph>, push: &T, image: Res<ImageView>,
	) {
		let mut pass = self.inner.start(
			pass,
			push,
			&[Attachment {
				image,
				load: Load::DontCare,
				store: true,
			}],
			None,
		);
		pass.draw(3, 1, 0, 0);
	}

	pub fn run<'a, 'frame, 'graph>(
		&self, pass: &'a mut PassContext<'frame, 'graph>, push: &T, attachments: &[Attachment],
	) {
		let mut pass = self.inner.start(pass, push, attachments, None);
		pass.draw(3, 1, 0, 0);
	}

	pub fn run_empty<'a, 'frame, 'graph>(
		&self, pass: &'a mut PassContext<'frame, 'graph>, push: &T, size: vk::Extent2D,
	) {
		let mut pass = self.inner.start_empty(pass, push, size);
		pass.draw(3, 1, 0, 0);
	}

	pub unsafe fn destroy(self) { self.inner.destroy(); }
}
