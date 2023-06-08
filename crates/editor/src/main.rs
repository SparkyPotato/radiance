use std::mem::ManuallyDrop;

use radiance_graph::{arena::Arena, device::Device, graph::RenderGraph, Result};
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, EnvFilter, Layer, Registry};
use winit::{
	event::{Event, WindowEvent},
	event_loop::{ControlFlow, EventLoop},
	window::WindowBuilder,
};

use crate::{ui::UiState, ui_handler::UiHandler, window::Window};

#[global_allocator]
static ALLOC: tracy::alloc::GlobalAllocator = tracy::alloc::GlobalAllocator::new();

mod ui;
mod ui_handler;
mod window;

struct State {
	device: Device,
	window: ManuallyDrop<Window>,
	graph: ManuallyDrop<RenderGraph>,
	ui: ManuallyDrop<UiHandler>,
}

impl State {
	pub fn new(event_loop: &EventLoop<()>, window: winit::window::Window) -> Result<Self> {
		let (device, surface) = unsafe { Device::with_window(&window, event_loop)? };
		let graph = ManuallyDrop::new(RenderGraph::new(&device)?);
		let window = ManuallyDrop::new(Window::new(&device, &graph, window, surface)?);
		let ui = ManuallyDrop::new(UiHandler::new(&device, event_loop, &window)?);

		Ok(Self {
			device,
			window,
			graph,
			ui,
		})
	}
}

impl Drop for State {
	fn drop(&mut self) {
		unsafe {
			ManuallyDrop::take(&mut self.graph).destroy(&self.device);
			ManuallyDrop::take(&mut self.ui).destroy(&self.device);
			ManuallyDrop::take(&mut self.window).destroy(&self.device);
		}
	}
}

fn main() {
	let _ = tracing::subscriber::set_global_default(
		Registry::default()
			.with(
				tracing_subscriber::fmt::layer()
					.pretty()
					.with_span_events(FmtSpan::CLOSE)
					.with_filter(EnvFilter::from_env("RADLOG")),
			)
			.with(tracy::tracing::TracyLayer),
	);

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("radiance-editor")
		.build(&event_loop)
		.unwrap();

	let mut state = State::new(&event_loop, window).unwrap();
	let mut ui = UiState::new(state.ui.fonts().clone());
	let mut arena = Arena::new();

	event_loop.run(move |event, _, flow| match event {
		Event::MainEventsCleared => state.window.request_redraw(),
		Event::RedrawRequested(_) => {
			arena.reset();
			let mut frame = state.graph.frame(&arena);

			let id = state
				.ui
				.run(&mut frame, &state.device, &state.window, |ctx| ui.render(ctx))
				.unwrap();

			frame.run(&state.device).unwrap();
			state.window.present(&state.device, id).unwrap();
			tracy::frame!();
		},
		Event::WindowEvent { event, .. } => {
			state.ui.on_event(&event);
			match event {
				WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
				WindowEvent::Resized(_) => state.window.resize(&state.device, &state.graph).unwrap(),
				_ => {},
			}
		},
		_ => {},
	})
}
