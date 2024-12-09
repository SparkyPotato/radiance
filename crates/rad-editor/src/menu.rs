use std::sync::Arc;

use rad_core::{asset::Asset, Engine};
use rad_ui::egui::{menu, Context, Key, KeyboardShortcut, Modifiers, TopBottomPanel};
use rad_world::World;
use rfd::FileDialog;

use crate::{asset::fs::FsAssetSystem, world::WorldContext};

pub struct Menu {}

impl Menu {
	pub fn new() -> Self { Self {} }

	pub fn render(&mut self, ctx: &Context, world: &mut WorldContext) {
		let fs: &Arc<FsAssetSystem> = Engine::get().asset_source().unwrap();

		let mut new = ctx.input_mut(|x| x.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::N)));
		let mut open = ctx.input_mut(|x| x.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::O)));

		TopBottomPanel::top("menu").show(ctx, |ui| {
			menu::bar(ui, |ui| {
				ui.menu_button("file", |ui| {
					new |= ui.button("new").clicked();
					open |= ui.button("open").clicked();
				});

				ui.menu_button("scene", |ui| {
					for w in fs.assets_of_type(World::uuid()) {
						if ui.button(w.to_string()).clicked() {
							let _ = world.open(w);
						}
					}
				})
			});
		});

		if new || open {
			if let Some(path) = FileDialog::new().pick_folder() {
				fs.open(path);
			}
		}
	}
}
