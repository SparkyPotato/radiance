use std::{path::PathBuf, sync::Arc};

use rad_core::{asset::Asset, Engine};
use rad_renderer::assets::mesh::Mesh;
use rad_ui::{
	egui::{Button, Context, Grid, Key, KeyboardShortcut, Modifiers, RichText, ScrollArea, TopBottomPanel},
	icons::{self, icon},
};
use rad_world::World;
use tracing::{error, info};

use crate::{
	asset::{fs::FsAssetSystem, import::GltfImporter},
	world::WorldContext,
};

pub mod fs;
mod import;

pub struct AssetTray {
	open: bool,
	cursor: PathBuf,
}

impl AssetTray {
	pub fn new() -> Self {
		Self {
			open: false,
			cursor: PathBuf::new(),
		}
	}

	pub fn render(&mut self, ctx: &Context, world: &mut WorldContext) {
		self.open =
			self.open ^ ctx.input_mut(|x| x.consume_shortcut(&KeyboardShortcut::new(Modifiers::COMMAND, Key::Space)));

		if self.open {
			TopBottomPanel::bottom("asset tray")
				.min_height(100.0)
				.resizable(true)
				.show(ctx, |ui| {
					let fs: &Arc<FsAssetSystem> = Engine::get().asset_source().unwrap();

					if fs.root().is_none() {
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
								ui.add(Button::new(icon(icons::PLUS)).frame(false));
							});

							ui.separator();

							ui.vertical(|ui| {
								ui.add_space(2.5);
								if ui
									.add(Button::new(icon(icons::ARROW_UP)).frame(false))
									.on_hover_text("back")
									.clicked()
								{
									self.cursor.pop();
								}
							});

							ui.separator();

							if ui
								.add(
									Button::new(fs.root().as_ref().unwrap().iter().last().unwrap().to_string_lossy())
										.frame(false),
								)
								.clicked()
							{
								self.cursor.clear();
							}

							for (i, x) in self.cursor.iter().enumerate() {
								ui.label("/");
								if ui.add(Button::new(x.to_string_lossy()).frame(false)).clicked() {
									self.cursor = self.cursor.components().take(i + 1).collect();
									break;
								}
							}
						});

						ui.add_space(5.0);

						let dir = fs.dir();
						let dir = dir.get_dir(&self.cursor).unwrap();
						let dirs = dir.dirs();
						let assets = dir.assets();
						let dir_count = dirs.len();
						let count = dir_count + assets.len();
						let rect = ui.available_rect_before_wrap();
						let width = rect.width();
						let per_row = (width / 60.0) as usize;
						let rows = count.div_ceil(per_row);
						ScrollArea::vertical()
							.auto_shrink([false, false])
							.drag_to_scroll(false)
							.show_rows(ui, 60.0, rows, |ui, range| {
								Grid::new("assets")
									.striped(false)
									.start_row(range.start)
									.min_col_width(60.0)
									.min_row_height(60.0)
									.max_col_width(60.0)
									.show(ui, |ui| {
										let start_item = range.start * per_row;
										let end_item = (range.end * per_row).min(count);

										let mut i = 0;
										for (n, _) in dirs.skip(start_item).take(end_item - start_item) {
											if i == per_row {
												ui.end_row();
												i = 0;
											}

											ui.vertical_centered(|ui| {
												if ui
													.add(Button::new(icon(icons::FOLDER).size(35.0)).frame(false))
													.double_clicked()
												{
													self.cursor.push(n.clone());
												}
												ui.label(n);
											});
										}

										let first_asset = start_item.saturating_sub(dir_count);
										for (n, header) in assets.skip(first_asset).take(end_item - first_asset) {
											if i == per_row {
												ui.end_row();
												i = 0;
											}

											let is_world = header.ty == World::uuid();
											ui.vertical_centered(|ui| {
												let i = if is_world {
													icons::MAP
												} else if header.ty == Mesh::uuid() {
													icons::CUBE
												} else {
													icons::FILE
												};
												if ui.add(Button::new(icon(i).size(35.0)).frame(false)).double_clicked()
													&& is_world
												{
													if let Err(e) = world.open(header.id) {
														error!("failed to open world: {:?}", e);
													}
												}
												ui.label(n);
											});
										}
									});
							});
					});
				});
		}
	}
}
