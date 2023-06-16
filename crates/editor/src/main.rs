use std::mem::ManuallyDrop;

use radiance_core::{CoreDevice, RenderCore};
use radiance_graph::{device::Device, graph::RenderGraph, Result};
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
	device: CoreDevice,
	graph: ManuallyDrop<RenderGraph>,
	core: ManuallyDrop<RenderCore>,
	state: ManuallyDrop<UiState>,
	ui: ManuallyDrop<UiHandler>,
	window: ManuallyDrop<Window>,
}

impl State {
	pub fn new(event_loop: &EventLoop<()>, window: winit::window::Window) -> Result<Self> {
		let (device, surface) = unsafe { Device::with_window(&window, event_loop)? };
		let device = CoreDevice::new(device)?;
		let graph = ManuallyDrop::new(RenderGraph::new(&device)?);
		let core = ManuallyDrop::new(RenderCore::new(
			&device,
			&[radiance_core::SHADERS, radiance_egui::SHADERS, radiance_passes::SHADERS],
		)?);
		let window = ManuallyDrop::new(Window::new(&device, &graph, window, surface)?);
		let ui = ManuallyDrop::new(UiHandler::new(&device, &core, event_loop, &window)?);
		let state = ManuallyDrop::new(UiState::new(&device, &core, ui.fonts().clone())?);

		Ok(Self {
			device,
			graph,
			core,
			ui,
			window,
			state,
		})
	}
}

impl Drop for State {
	fn drop(&mut self) {
		unsafe {
			let _ = self.device.device().device_wait_idle();

			ManuallyDrop::take(&mut self.graph).destroy(&self.device);
			ManuallyDrop::take(&mut self.state).destroy(&self.device);
			ManuallyDrop::take(&mut self.core).destroy(&self.device);
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

	event_loop.run(move |event, _, flow| match event {
		Event::MainEventsCleared => state.window.request_redraw(),
		Event::RedrawRequested(_) => {
			let mut frame = state.core.frame(&state.device, &mut state.graph).unwrap();

			state.ui.begin_frame(&state.window);
			state.state.render(&state.device, &mut frame, &state.ui.ctx);
			let id = state.ui.run(&state.device, &mut frame, &state.window).unwrap();

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
