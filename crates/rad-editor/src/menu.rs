use std::sync::Arc;

use rad_core::Engine;
use rad_ui::egui::{menu, Context, Key, KeyboardShortcut, Modifiers, TopBottomPanel};
use rfd::FileDialog;

use crate::{asset::fs::FsAssetSystem, render::Renderer};

pub struct Menu {}

impl Menu {
	pub fn new() -> Self { Self {} }

	pub fn render(&mut self, ctx: &Context, renderer: &mut Renderer) {
		let fs: &Arc<FsAssetSystem> = Engine::get().asset_source();

		let mut new = ctx.input_mut(|x| x.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::N)));
		let mut open = ctx.input_mut(|x| x.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::O)));

		TopBottomPanel::top("menu").show(ctx, |ui| {
			menu::bar(ui, |ui| {
				ui.menu_button("file", |ui| {
					new |= ui.button("new").clicked();
					open |= ui.button("open").clicked();
				});

				ui.menu_button("window", |ui| {
					ui.checkbox(&mut renderer.debug_window.enabled, "debug");
				});
			});
		});

		if new || open {
			if let Some(path) = FileDialog::new().pick_folder() {
				fs.open(path);
			}
		}
	}
}
