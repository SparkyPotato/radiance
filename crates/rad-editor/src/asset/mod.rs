use std::sync::Arc;

use rad_core::Engine;
use rad_ui::{
	egui::{Context, Key, KeyboardShortcut, Modifiers, RichText, TopBottomPanel},
	icons,
	widgets::icon_button::UiIconButton,
};
use tracing::{error, info};

use crate::asset::{fs::FsAssetSystem, import::GltfImporter};

pub mod fs;
mod import;

pub struct AssetTray {
	open: bool,
}

impl AssetTray {
	pub fn new() -> Self { Self { open: false } }

	pub fn render(&mut self, ctx: &Context) {
		self.open =
			self.open ^ ctx.input_mut(|x| x.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Space)));

		if self.open {
			TopBottomPanel::bottom("asset tray")
				.min_height(100.0)
				.resizable(true)
				.show(ctx, |ui| {
					if !Engine::get().asset_source::<Arc<FsAssetSystem>>().unwrap().is_open() {
						ui.centered_and_justified(|ui| {
							ui.label(RichText::new("no project opened").size(20.0));
						});
						return;
					} else if ctx.input(|x| !x.raw.hovered_files.is_empty()) {
						ui.centered_and_justified(|ui| {
							ui.label(RichText::new("drop files to import").size(20.0));
						});
						return;
					}

					let dropped = ctx.input_mut(|x| std::mem::take(&mut x.raw.dropped_files));
					for file in dropped {
						if let Some(x) = GltfImporter::initialize(&file.path.unwrap()) {
							if let Err(e) = x.and_then(|x| {
								x.import(|x| {
									info!("import: {:.2}%", x * 100.0);
								})
							}) {
								error!("import error: {:?}", e);
							}
						}
					}

					ui.vertical(|ui| {
						ui.add_space(5.0);
						ui.horizontal(|ui| {
							ui.vertical(|ui| {
								ui.add_space(2.5);
								ui.icon_button(icons::PLUS);
							});

							ui.separator();

							ui.vertical(|ui| {
								ui.add_space(2.5);
								ui.icon_button(icons::ARROW_UP);
							});

							ui.separator();
						});
					});
				});
		}
	}
}
