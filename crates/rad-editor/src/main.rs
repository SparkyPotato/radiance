use rad_core::{Engine, EngineBuilder, Module};
use rad_graph::{graph::Frame, Result};
use rad_renderer::RendererModule;
use rad_rhi::RhiModule;
use rad_ui::{
	egui::{CentralPanel, Context},
	App,
	UiApp,
	UiModule,
};
use rad_window::WindowModule;
use rad_world::WorldModule;
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, EnvFilter, Layer, Registry};

use crate::{
	asset::{fs::FsAssetSystem, AssetTray},
	menu::Menu,
};

mod asset;
mod menu;

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

	rad_window::run(UiApp::new(EditorApp {
		menu: Menu::new(),
		assets: AssetTray::new(),
	})?)
}

struct EditorModule;

impl Module for EditorModule {
	fn init(engine: &mut EngineBuilder) { engine.asset_source(FsAssetSystem::new()); }
}

struct EditorApp {
	menu: Menu,
	assets: AssetTray,
}

impl App for EditorApp {
	fn render<'pass>(&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context) -> Result<()> {
		self.menu.render(ctx);
		self.assets.render(ctx);
		CentralPanel::default().show(ctx, |ui| ui.label("hi"));

		Ok(())
	}
}
