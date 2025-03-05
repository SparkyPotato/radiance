#![feature(path_add_extension)]

use std::mem::ManuallyDrop;

use rad_core::{Engine, EngineBuilder, Module};
use rad_graph::{graph::Frame, Result};
use rad_renderer::RendererModule;
use rad_rhi::RhiModule;
use rad_ui::{egui::Context, App, UiApp, UiModule, Window};
use rad_window::{
	winit::{event::WindowEvent, window::Window as WWindow},
	WindowModule,
};
use rad_world::WorldModule;
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, EnvFilter, Layer, Registry};

use crate::{
	asset::{fs::FsAssetSystem, AssetTray},
	menu::Menu,
	render::Renderer,
	world::WorldContext,
};

mod asset;
mod menu;
mod render;
mod world;

fn main() -> Result<()> {
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

	Engine::builder()
		.module::<RhiModule>()
		.module::<UiModule>()
		.module::<WindowModule>()
		.module::<WorldModule>()
		.module::<RendererModule>()
		.module::<EditorModule>()
		.build();

	rad_window::run(UiApp::new(EditorApp::new())?)
}

struct EditorModule;

impl Module for EditorModule {
	fn init(engine: &mut EngineBuilder) { engine.asset_source(FsAssetSystem::new()); }
}

struct EditorApp {
	menu: Menu,
	assets: AssetTray,
	world: WorldContext,
	renderer: ManuallyDrop<Renderer>,
}

impl EditorApp {
	fn new() -> Self {
		Self {
			menu: Menu::new(),
			assets: AssetTray::new(),
			world: WorldContext::new(),
			renderer: ManuallyDrop::new(Renderer::new().unwrap()),
		}
	}
}

impl App for EditorApp {
	fn render<'pass>(&'pass mut self, window: &Window, frame: &mut Frame<'pass, '_>, ctx: &Context) -> Result<()> {
		self.menu.render(ctx, &mut self.renderer);
		self.assets.render(ctx, &mut self.world);
		self.renderer.render(window, frame, ctx, &mut self.world);

		Ok(())
	}

	fn on_window_event(&mut self, window: &WWindow, event: &WindowEvent) {
		self.renderer.on_window_event(window, event);
	}
}

impl Drop for EditorApp {
	fn drop(&mut self) {
		unsafe {
			ManuallyDrop::take(&mut self.renderer).destroy();
		}
	}
}
