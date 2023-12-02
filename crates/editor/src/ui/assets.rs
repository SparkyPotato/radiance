use std::path::PathBuf;

use egui::{Context, RichText, ScrollArea};
use egui_extras::{Size, Strip, StripBuilder};
use radiance_asset::fs::FsSystem;
use tracing::{event, Level};

use crate::ui::{
	widgets::{icons, IntoIcon, UiExt},
	Fonts,
};

#[derive(Default)]
pub struct AssetManager {
	pub system: Option<FsSystem>,
	cursor: PathBuf,
}

impl AssetManager {
	pub fn open(&mut self, path: impl Into<PathBuf>) {
		let buf = std::fs::canonicalize(path.into()).unwrap();
		self.system = Some(FsSystem::new(buf.clone()));
		self.cursor = buf;
	}

	pub fn render(&mut self, ctx: &Context, fonts: &Fonts) {
		egui::TopBottomPanel::bottom("assets")
			.resizable(true)
			.min_height(100.0)
			.show(ctx, |ui| {
				if let Some(ref mut sys) = self.system {
					if ctx.input(|x| !x.raw.hovered_files.is_empty()) {
						ui.centered_and_justified(|ui| {
							ui.label(RichText::new("drop files to import").size(20.0));
						});
					} else {
						let dropped = ctx.input_mut(|x| std::mem::take(&mut x.raw.dropped_files));
						for file in dropped {
							let path = file.path.unwrap();
							if let Err(err) = sys.import(&path, &self.cursor, |x, y| {
								event!(Level::INFO, "{:.2}%", x.as_percentage(y))
							}) {
								event!(Level::ERROR, "failed to import {:?}: {:?}", path, err);
							}
						}

						ui.vertical(|ui| {
							ui.add_space(5.0);
							ui.horizontal(|ui| {
								ui.vertical(|ui| {
									ui.add_space(2.5);
									if ui
										.text_button(fonts.icons.text(icons::ARROW_UP).heading())
										.on_hover_text("Go back")
										.clicked() && self.cursor != sys.root()
									{
										self.cursor.pop();
									}
								});

								ui.separator();

								let root = sys.root().parent().unwrap();
								let current = self.cursor.strip_prefix(root).unwrap();
								for (i, component) in current.components().enumerate() {
									if ui
										.text_button(RichText::new(component.as_os_str().to_string_lossy()).heading())
										.clicked()
									{
										self.cursor = root.join(current.components().take(i + 1).collect::<PathBuf>());
										break;
									}
									ui.label(RichText::new("/").heading());
								}
							});
							ui.add_space(5.0);

							let rect = ui.available_rect_before_wrap();
							const CELL_SIZE: f32 = 50.0;
							let width = rect.width();
							let height = rect.height();
							let cells_x = (width / CELL_SIZE) as usize;
							let cells_y = (height / CELL_SIZE) as usize;

							let view = sys.dir_view(&self.cursor).unwrap();
							ScrollArea::vertical()
								.min_scrolled_width(width)
								.min_scrolled_height(height)
								.auto_shrink([false, false])
								.stick_to_right(true)
								.show(ui, |ui| {
									let mut dirs = view.dirs();
									let mut assets = view.assets();
									let mut make_cell = |strip: &mut Strip| match dirs.next() {
										Some(dir) => strip.cell(|ui| {
											ui.vertical(|ui| {
												ui.centered_and_justified(|ui| {
													let name = dir.name();
													if ui
														.text_button(fonts.icons.text(icons::FOLDER).size(32.0))
														.clicked()
													{
														self.cursor.push(name);
													}
													ui.label(RichText::new(name).heading());
												});
											});
										}),
										None => match assets.next() {
											Some((asset, _)) => strip.cell(|ui| {
												ui.vertical(|ui| {
													ui.centered_and_justified(|ui| {
														ui.label(fonts.icons.text(icons::FILE).size(32.0));
														ui.label(RichText::new(asset).heading());
													});
												});
											}),
											None => strip.empty(),
										},
									};

									StripBuilder::new(ui).sizes(Size::exact(CELL_SIZE), cells_y).vertical(
										|mut strip| {
											for _ in 0..cells_y {
												strip.strip(|builder| {
													builder.sizes(Size::exact(CELL_SIZE), cells_x).horizontal(
														|mut strip| {
															for _ in 0..cells_x {
																make_cell(&mut strip);
															}
														},
													);
												});
											}
										},
									);
								});
						});
					}
				} else {
					ui.centered_and_justified(|ui| {
						ui.label(RichText::new("no project loaded").size(20.0));
					});
				}
			});
	}
}
