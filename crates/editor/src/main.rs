use std::mem::ManuallyDrop;

use ash::{ext, vk};
use radiance_graph::{arena::Arena, device::Device, graph::RenderGraph, Result};
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, EnvFilter, Layer, Registry};
use winit::{
	dpi::LogicalSize,
	event::{Event, StartCause, WindowEvent},
	event_loop::EventLoop,
	window::WindowBuilder,
};

use crate::{ui::UiState, ui_handler::UiHandler, window::Window};

#[global_allocator]
static ALLOC: tracy::alloc::GlobalAllocator = tracy::alloc::GlobalAllocator::new();

mod ui;
mod ui_handler;
mod window;

fn init_device(window: &winit::window::Window, event_loop: &EventLoop<()>) -> Result<(Device, vk::SurfaceKHR)> {
	unsafe {
		// TODO: Move features somewhere else.
		Device::builder()
			.window(window, event_loop)
			.device_extensions(&[ext::mesh_shader::NAME, ext::shader_image_atomic_int64::NAME])
			.features(
				vk::PhysicalDeviceFeatures2::default()
					.features(
						vk::PhysicalDeviceFeatures::default()
							.shader_int16(true)
							.shader_int64(true)
							.fragment_stores_and_atomics(true),
					)
					.push_next(
						&mut vk::PhysicalDeviceVulkan11Features::default()
							.variable_pointers(true)
							.variable_pointers_storage_buffer(true)
							.storage_push_constant16(true)
							.storage_buffer16_bit_access(true),
					)
					.push_next(
						&mut vk::PhysicalDeviceVulkan12Features::default()
							.sampler_filter_minmax(true)
							.shader_int8(true)
							.storage_buffer8_bit_access(true)
							.storage_push_constant8(true)
							.scalar_block_layout(true),
					)
					.push_next(
						&mut vk::PhysicalDeviceVulkan13Features::default()
							.dynamic_rendering(true)
							.shader_demote_to_helper_invocation(true),
					)
					.push_next(&mut vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true))
					.push_next(
						&mut vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default()
							.shader_image_int64_atomics(true),
					),
			)
			.build()
	}
}

struct State {
	device: Device,
	arena: Arena,
	graph: ManuallyDrop<RenderGraph>,
	state: ManuallyDrop<UiState>,
	ui: ManuallyDrop<UiHandler>,
	window: ManuallyDrop<Window>,
}

impl State {
	pub fn new(event_loop: &EventLoop<()>, window: winit::window::Window) -> Result<Self> {
		let (device, surface) = init_device(&window, event_loop)?;
		let graph = ManuallyDrop::new(RenderGraph::new(&device)?);
		let window = ManuallyDrop::new(Window::new(&device, window, surface)?);
		let ui = ManuallyDrop::new(UiHandler::new(&device, event_loop, &window)?);
		let state = ManuallyDrop::new(UiState::new(&device, ui.fonts().clone())?);

		Ok(Self {
			device,
			arena: Arena::new(),
			graph,
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

	let event_loop = EventLoop::new().unwrap();
	let window = WindowBuilder::new()
		.with_title("radiance-editor")
		.with_inner_size(LogicalSize::new(1280, 720))
		.with_visible(false)
		.build(&event_loop)
		.unwrap();

	let mut state = State::new(&event_loop, window).unwrap();

	event_loop
		.run(move |event, t| match event {
			Event::NewEvents(StartCause::Init) => state.window.window.set_visible(true),
			Event::AboutToWait => state.window.request_redraw(),
			Event::WindowEvent { event, .. } => {
				match event {
					WindowEvent::RedrawRequested => {
						let arena_size = state.arena.memory_usage();
						state.arena.reset();
						let mut frame = state.graph.frame(&state.device, &state.arena);
						let id = {
							tracy::zone!("pass generation");
							state.ui.begin_frame(&state.window);
							state.state.render(&mut frame, &state.ui.ctx, &state.window, arena_size);
							state.ui.run(&mut frame, &mut state.window).unwrap()
						};

						frame.run().unwrap();
						state.window.present(&state.device, id).unwrap();
						tracy::frame!();
					},
					WindowEvent::CloseRequested => t.exit(),
					WindowEvent::Resized(_) => state.window.resize(&state.device).unwrap(),
					_ => {},
				}
				state.state.on_window_event(&state.window, &event);
				state.ui.on_event(&state.window, &event);
			},
			_ => {},
		})
		.unwrap()
}
