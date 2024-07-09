use std::{io, path::PathBuf, sync::Arc, time::Instant};

use crossbeam_channel::{Receiver, Sender};
use egui::{Align, Context, Label, Layout, ProgressBar, RichText, ScrollArea, Ui};
use egui_extras::Size;
use egui_grid::GridBuilder;
use radiance_asset::{
	fs::FsSystem,
	import::{ImportError, ImportProgress},
	AssetType,
};
use tracing::{event, Level};

use crate::ui::{
	notif::{NotifStack, NotifType, Notification, PushNotif},
	render::Renderer,
	widgets::{icons, IntoIcon, UiExt},
	Fonts,
};

struct ImportNotif {
	recv: Receiver<Progress>,
	latest: Progress,
	finished: Option<Instant>,
}

impl Notification for ImportNotif {
	fn draw(&mut self, ui: &mut Ui, _: &Fonts) {
		for progress in self.recv.try_iter() {
			self.latest = progress;
		}

		match &self.latest {
			Progress::Discovering => {
				ui.label("discovering");
			},
			Progress::Update { done, total } => {
				ui.add(ProgressBar::new(done.as_percentage(*total)).show_percentage());
			},
			Progress::Finished => {
				if self.finished.is_none() {
					self.finished = Some(Instant::now());
				}
				ui.label("imported");
			},
			Progress::Error(err) => {
				ui.label(format!("failed: {:?}", err));
			},
		}
	}

	fn expired(&self) -> bool { self.finished.map(|x| x.elapsed().as_secs_f32() > 5.0).unwrap_or(false) }

	fn dismissable(&self) -> bool { matches!(self.latest, Progress::Error(_) | Progress::Finished) }
}

enum Progress {
	Discovering,
	Update {
		done: ImportProgress,
		total: ImportProgress,
	},
	Finished,
	Error(ImportError<io::Error, io::Error>),
}

struct ImportRequest {
	from: PathBuf,
	to: PathBuf,
	progress: Sender<Progress>,
}

enum ThreadMsg {
	Import(ImportRequest),
	Open(PathBuf),
}

enum ThreadRecv {
	OpenedSystem { sys: Arc<FsSystem>, existed: bool },
}

fn import_thread(recv: Receiver<ThreadMsg>, send: Sender<ThreadRecv>) {
	tracy::set_thread_name(tracy::c_str!("importer"));
	let mut sys = None;
	for msg in recv {
		match msg {
			ThreadMsg::Open(path) => {
				let existed = path.exists();
				let s = Arc::new(FsSystem::new(path));
				let _ = send.send(ThreadRecv::OpenedSystem {
					sys: s.clone(),
					existed,
				});
				sys = Some(s);
			},
			ThreadMsg::Import(req) => {
				if let Some(ref sys) = sys {
					if let Err(err) = sys.import(&req.from, &req.to, |x, y| {
						event!(Level::INFO, "{:.2}%", x.as_percentage(y) * 100.0);
						let _ = req.progress.send(Progress::Update { done: x, total: y });
					}) {
						event!(Level::ERROR, "failed to import {:?}: {:?}", req.from, err);
						let _ = req.progress.send(Progress::Error(err));
					}
				}
				let _ = req.progress.send(Progress::Finished);
			},
		}
	}
}

pub struct AssetManager {
	pub system: Option<Arc<FsSystem>>,
	cursor: PathBuf,
	send: Sender<ThreadMsg>,
	recv: Receiver<ThreadRecv>,
}

impl AssetManager {
	pub fn new() -> Self {
		let (send, recv) = crossbeam_channel::unbounded();
		let (isend, irecv) = crossbeam_channel::unbounded();
		std::thread::Builder::new()
			.name("importer".to_string())
			.spawn(move || import_thread(recv, isend))
			.unwrap();
		Self {
			system: None,
			cursor: PathBuf::new(),
			send,
			recv: irecv,
		}
	}

	pub fn open(&mut self, path: impl Into<PathBuf>) {
		let buf = std::fs::canonicalize(path.into()).unwrap();
		self.send.send(ThreadMsg::Open(buf.clone())).unwrap();
		self.cursor = buf;
	}

	pub fn render(&mut self, ctx: &Context, notifs: &mut NotifStack, renderer: &mut Renderer, fonts: &Fonts) {
		for msg in self.recv.try_iter() {
			match msg {
				ThreadRecv::OpenedSystem { sys, existed } => {
					self.system = Some(sys);
					notifs.push(
						NotifType::Info,
						"project",
						PushNotif::new(format!("{} project", if existed { "loaded" } else { "created" })),
					);
				},
			}
		}

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
							let from = file.path.unwrap();
							let (progress, recv) = crossbeam_channel::unbounded();
							let _ = self.send.send(ThreadMsg::Import(ImportRequest {
								from,
								to: self.cursor.clone(),
								progress,
							}));
							notifs.push(
								NotifType::Info,
								"import",
								ImportNotif {
									recv,
									latest: Progress::Discovering,
									finished: None,
								},
							);
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

							let view = sys.dir_view(&self.cursor).unwrap();
							let dirs = view.dir_count();
							let assets = view.asset_count();
							let count = dirs + assets;

							let rect = ui.available_rect_before_wrap();
							const CELL_SIZE: f32 = 75.0;
							let width = rect.width();
							let height = rect.height();
							let cells_x = (width / CELL_SIZE) as usize - 1;
							let rows = (count + cells_x - 1) / cells_x;

							ScrollArea::vertical()
								.min_scrolled_width(width)
								.min_scrolled_height(height)
								.auto_shrink([false, false])
								.stick_to_right(true)
								.show_rows(ui, CELL_SIZE, rows, |ui, range| {
									let mut grid = GridBuilder::new().layout_standard(Layout::top_down(Align::Center));
									for _ in range.clone() {
										grid = grid.new_row(Size::remainder());
										grid = grid.cells(Size::exact(CELL_SIZE), cells_x as _);
									}

									let start_obj = range.start * cells_x;
									let end_obj = range.end * cells_x;
									let obj_range = start_obj..end_obj;

									grid.show(ui, |mut grid| {
										let mut i = 0;
										view.for_each_dir(|dir| {
											if !obj_range.contains(&i) {
												i += 1;
												return;
											}

											grid.cell(|ui| {
												let name = dir.name();
												if ui
													.text_button(fonts.icons.text(icons::FOLDER).size(32.0))
													.double_clicked()
												{
													self.cursor.push(&*name);
												}
												ui.add(Label::new(&*name).truncate(true));
											});
											i += 1;
										});

										view.for_each_asset(|name, uuid| {
											if !obj_range.contains(&i) {
												i += 1;
												return;
											}

											grid.cell(|ui| {
												if ui
													.text_button(fonts.icons.text(icons::FILE).size(32.0))
													.double_clicked()
												{
													if sys.metadata(uuid).unwrap().ty == AssetType::Scene {
														renderer.set_scene(uuid);
													}
												}
												ui.add(Label::new(name).truncate(true));
											});
											i += 1;
										})
									});
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
