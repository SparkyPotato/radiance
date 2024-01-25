//! The entire editor UI.

mod assets;
mod notif;
mod render;
mod widgets;

use egui::{menu, Context, TopBottomPanel};
use radiance_core::{CoreDevice, CoreFrame, RenderCore};
use radiance_graph::Result;
use rfd::FileDialog;
pub use widgets::Fonts;
use winit::event::WindowEvent;

use crate::{
	ui::{assets::AssetManager, notif::NotifStack, render::Renderer},
	window::Window,
};

pub struct UiState {
	assets: AssetManager,
	fonts: Fonts,
	renderer: Renderer,
	notifs: NotifStack,
}

impl UiState {
	pub fn new(device: &CoreDevice, core: &RenderCore, fonts: Fonts) -> Result<Self> {
		Ok(Self {
			fonts,
			assets: AssetManager::new(),
			renderer: Renderer::new(device, core)?,
			notifs: NotifStack::new(),
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

				ui.menu_button("debug", |ui| self.renderer.draw_debug_menu(ui));

				ui.menu_button("cameras", |ui| self.renderer.draw_camera_menu(ui));
			});
		});

		self.renderer.draw_debug_windows(ctx, device);

		self.assets
			.render(ctx, &mut self.notifs, &mut self.renderer, frame, &self.fonts);
		self.renderer
			.render(device, frame, ctx, window, self.assets.system.as_deref().map(|x| &**x));

		self.notifs.render(ctx, &self.fonts);
	}

	pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
		self.renderer.on_window_event(window, event);
	}

	pub unsafe fn destroy(self, device: &CoreDevice) { self.renderer.destroy(device); }
}

