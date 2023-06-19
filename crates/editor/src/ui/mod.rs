//! The entire editor UI.

mod assets;
mod render;
mod widgets;

use egui::{menu, Context, TopBottomPanel};
use radiance_asset::AssetType;
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_graph::Result;
use rfd::FileDialog;
pub use widgets::Fonts;

use crate::{
	ui::{assets::AssetManager, render::Renderer},
	window::Window,
};

pub struct UiState {
	assets: AssetManager,
	fonts: Fonts,
	renderer: Renderer,
}

impl UiState {
	pub fn new(device: &CoreDevice, core: &RenderCore, fonts: Fonts) -> Result<Self> {
		Ok(Self {
			fonts,
			assets: AssetManager::default(),
			renderer: Renderer::new(device, core)?,
		})
	}

	pub fn render<'pass>(
		&'pass mut self, device: &CoreDevice, frame: &mut CoreFrame<'pass, '_>, ctx: &Context, window: &Window,
	) {
		TopBottomPanel::top("menu").show(ctx, |ui| {
			menu::bar(ui, |ui| {
				ui.menu_button("project", |ui| {
					let new = ui.button("new").clicked();
					let load = ui.button("load").clicked();
					if new || load {
						if let Some(path) = FileDialog::new().pick_folder() {
							self.assets.open(path);
						}
					}
				});

				ui.menu_button("render", |ui| {
					ui.menu_button("scene", |ui| {
						for asset in self
							.assets
							.system
							.as_ref()
							.into_iter()
							.flat_map(|x| x.assets_of_type(AssetType::Scene))
						{
							if ui.button(format!("{}", asset)).clicked() {
								self.renderer.set_scene(frame.ctx(), asset);
							}
						}
					});
				});
			});
		});

		self.assets.render(ctx, &self.fonts);
		self.renderer
			.render(device, frame, ctx, window, self.assets.system.as_deref_mut());
	}

	pub unsafe fn destroy(self, device: &CoreDevice) { self.renderer.destroy(device); }
}
