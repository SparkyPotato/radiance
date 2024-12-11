#![feature(allocator_api)]

use std::mem::ManuallyDrop;

pub use egui;
use egui::{Context, ViewportId};
use egui_winit::{pixels_per_point, winit::window::Theme, State};
use rad_core::{Engine, EngineBuilder, Module};
use rad_graph::{
	arena::Arena,
	graph::{Frame, RenderGraph, SwapchainImage},
	Result,
};
use rad_window::winit::{event::WindowEvent, event_loop::ActiveEventLoop, window::Window};
use vek::Vec2;

pub use crate::render::{raw_texture_to_id, to_texture_id};
use crate::render::{Renderer, ScreenDescriptor};

mod fonts;
pub mod icons;
mod render;
pub mod widgets;

pub struct UiModule;

impl Module for UiModule {
	fn init(engine: &mut EngineBuilder) {
		let ctx = Context::default();
		fonts::setup_fonts(&ctx);
		engine.global(ctx);
	}
}

pub struct UiApp<T> {
	inner: T,
	arena: Arena,
	graph: ManuallyDrop<RenderGraph>,
	renderer: ManuallyDrop<Renderer>,
	state: Option<State>,
}

pub trait App {
	fn render<'pass>(&'pass mut self, window: &Window, frame: &mut Frame<'pass, '_>, ctx: &Context) -> Result<()>;

	fn on_window_event(&mut self, _window: &Window, _event: &WindowEvent) {}
}

impl<T: App> UiApp<T> {
	pub fn new(inner: T) -> Result<Self> {
		let arena = Arena::new();
		let graph = RenderGraph::new(Engine::get().global())?;
		let renderer = Renderer::new(Engine::get().global())?;
		Ok(Self {
			inner,
			arena,
			graph: ManuallyDrop::new(graph),
			renderer: ManuallyDrop::new(renderer),
			state: None,
		})
	}
}

impl<T: App> rad_window::App for UiApp<T> {
	fn init(&mut self, el: &ActiveEventLoop, _: &Window) -> Result<()> {
		self.state = Some(State::new(
			Engine::get().global::<Context>().clone(),
			ViewportId::default(),
			el,
			None,
			Some(Theme::Dark),
			None,
		));

		Ok(())
	}

	fn draw(&mut self, window: &Window, image: SwapchainImage) -> Result<()> {
		let ctx = Engine::get().global::<Context>();
		self.arena.reset();

		let mut frame = self.graph.frame(Engine::get().global(), &self.arena);

		ctx.begin_pass(self.state.as_mut().unwrap().take_egui_input(window));
		self.inner.render(window, &mut frame, ctx)?;
		let output = ctx.end_pass();

		self.state
			.as_mut()
			.unwrap()
			.handle_platform_output(window, output.platform_output);
		let tris = ctx.tessellate(output.shapes, pixels_per_point(&ctx, window));

		self.renderer.run(
			&mut frame,
			tris,
			output.textures_delta,
			ScreenDescriptor {
				physical_size: Vec2::new(window.inner_size().width, window.inner_size().height),
				scaling: window.scale_factor() as _,
			},
			image,
		);

		frame.run()?;

		tracy::frame!();

		Ok(())
	}

	fn event(&mut self, window: &Window, event: WindowEvent) -> Result<()> {
		self.inner.on_window_event(window, &event);
		let _ = self.state.as_mut().unwrap().on_window_event(window, &event);
		Ok(())
	}
}

impl<T> Drop for UiApp<T> {
	fn drop(&mut self) {
		unsafe {
			ManuallyDrop::take(&mut self.graph).destroy(Engine::get().global());
			ManuallyDrop::take(&mut self.renderer).destroy(Engine::get().global());
		}
	}
}
