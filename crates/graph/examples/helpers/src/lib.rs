use std::{
	mem::ManuallyDrop,
	time::{Duration, Instant},
};

pub use bytemuck;
pub use naga::ShaderStage;
use radiance_graph::{
	arena::Arena,
	ash::vk::Format,
	device::Device,
	graph::{ExternalImage, Frame, RenderGraph},
};
use tracing_subscriber::fmt;
pub use vek;
use vek::Vec2;
use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

use crate::swapchain::Swapchain;

pub mod cmd;
pub mod load;
pub mod misc;
pub mod pipeline;

mod swapchain;

pub struct RenderInput<'a> {
	pub image: ExternalImage<'a>,
	pub format: Format,
	pub size: Vec2<u32>,
}

pub trait App: 'static + Sized {
	const NAME: &'static str;

	fn create(device: &Device) -> Self;

	fn destroy(self, device: &Device);

	fn render<'frame>(&'frame mut self, frame: &mut Frame<'frame, '_>, input: RenderInput, dt: Duration);
}

pub fn run<T: App>() -> ! {
	let _ = tracing::subscriber::set_global_default(fmt().pretty().finish());

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title(format!("Example: {}", T::NAME))
		.build(&event_loop)
		.unwrap();

	let mut arena = Arena::new();

	let (device, surface) = unsafe { Device::with_window(&window, &event_loop).unwrap() };
	let mut swapchain = ManuallyDrop::new(Swapchain::new(&device, surface, &window));
	let mut graph = ManuallyDrop::new(RenderGraph::new(&device).unwrap());
	let mut app = ManuallyDrop::new(T::create(&device));

	let mut prev = Instant::now();
	event_loop.run(move |event, _, flow| match event {
		Event::MainEventsCleared => window.request_redraw(),
		Event::RedrawRequested(_) => {
			arena.reset();
			let mut frame = graph.frame(&arena);

			let (image, id) = swapchain.acquire();

			let size = window.inner_size();
			app.render(
				&mut frame,
				RenderInput {
					image,
					format: Format::B8G8R8A8_SRGB,
					size: Vec2::new(size.width, size.height),
				},
				prev.elapsed(),
			);
			prev = Instant::now();
			frame.run(&device).unwrap();

			swapchain.present(&device, id, &graph);
		},
		Event::WindowEvent { event, .. } => match event {
			WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
			WindowEvent::Resized(_) => swapchain.resize(surface, &window, &graph),
			_ => {},
		},
		Event::LoopDestroyed => unsafe {
			ManuallyDrop::take(&mut graph).destroy(&device);
			ManuallyDrop::take(&mut app).destroy(&device);
			ManuallyDrop::take(&mut swapchain).destroy(&device);
			device.surface_ext().unwrap().destroy_surface(surface, None);
		},
		_ => {},
	})
}
