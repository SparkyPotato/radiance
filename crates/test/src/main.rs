use std::{mem::ManuallyDrop, time::Instant};

use radiance_graph::{arena::Arena, device::Device, graph::RenderGraph};
use tracing_subscriber::{fmt, fmt::format::FmtSpan, layer::SubscriberExt, EnvFilter};
use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

use crate::{ui_handler::UiHandler, window::Window};

#[global_allocator]
static ALLOC: tracy::alloc::GlobalAllocator = tracy::alloc::GlobalAllocator::new();

mod ui_handler;
mod window;

fn main() {
	let _ = tracing::subscriber::set_global_default(
		fmt()
			.pretty()
			.with_env_filter(EnvFilter::from_env("RADLOG"))
			.with_span_events(FmtSpan::CLOSE)
			.finish()
			.with(tracy::tracing::TracyLayer),
	);

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("radiance-test")
		.build(&event_loop)
		.unwrap();
	let mut ui = UiHandler::new(&event_loop);

	let (device, surface) = unsafe { Device::with_window(&window, &event_loop).unwrap() };
	let mut window = ManuallyDrop::new(Window::new(&device, window, surface));
	let mut graph = ManuallyDrop::new(RenderGraph::new(&device).unwrap());
	let mut arena = Arena::new();

	let mut prev = Instant::now();
	event_loop.run(move |event, _, flow| match event {
		Event::MainEventsCleared => window.request_redraw(),
		Event::RedrawRequested(_) => {
			let dt = prev.elapsed();
			prev = Instant::now();

			arena.reset();
			let mut frame = graph.frame(&arena);
			let (image, id) = window.acquire();

			ui.run(&mut frame, image, &window, |ctx| {
				egui::Window::new("sus").show(&ctx, |ui| {
					ui.label("sus");
				});
			});

			frame.run(&device).unwrap();
			window.present(&device, id, &graph);
			tracy::frame!();
		},
		Event::WindowEvent { event, .. } => {
			ui.on_event(&event);
			match event {
				WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
				WindowEvent::Resized(_) => window.resize(&device, &graph),
				_ => {},
			}
		},
		Event::LoopDestroyed => unsafe {
			ManuallyDrop::take(&mut graph).destroy(&device);
			ManuallyDrop::take(&mut window).destroy(&device);
			device.surface_ext().unwrap().destroy_surface(surface, None);
		},
		_ => {},
	})
}
