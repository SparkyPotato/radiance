//! The entire editor UI.

mod assets;
mod widgets;

use std::sync::Arc;

use egui::{menu, CentralPanel, Context, TopBottomPanel};
use rfd::FileDialog;

use crate::ui::assets::AssetManager;

pub struct UiState {
	assets: AssetManager,
	icon: Arc<str>,
}

impl UiState {
	pub fn new(icon: Arc<str>) -> Self {
		Self {
			assets: AssetManager::default(),
			icon,
		}
	}

	pub fn render(&mut self, ctx: &Context) {
		TopBottomPanel::top("menu").show(ctx, |ui| {
			menu::bar(ui, |ui| {
				ui.menu_button("Project", |ui| {
					let new = ui.button("New").clicked();
					let load = ui.button("Load").clicked();
					if new || load {
						if let Some(path) = FileDialog::new().pick_folder() {
							self.assets.open(path);
						}
					}
				})
			});
		});

		self.assets.render(ctx, &self.icon);

		CentralPanel::default().show(ctx, |ui| {});
	}
}
