use std::{io, path::PathBuf, sync::Arc, time::Instant};

use crossbeam_channel::{Receiver, Sender};
use egui::{Align, Context, Label, Layout, ProgressBar, RichText, ScrollArea, Ui};
use egui_extras::Size;
use egui_grid::GridBuilder;
use radiance_asset::{gltf::GltfImporter, mesh::Mesh, scene::Scene, Asset, AssetSystem, DirView, Importer};
use radiance_graph::graph::Frame;
use tracing::{event, Level};

use crate::ui::{
	notif::{NotifStack, NotifType, Notification, PushNotif},
	render::Renderer,
	widgets::{icons, IntoIcon, UiExt},
	Fonts,
};

enum Progress {
	Discovering,
	Update(f32),
	Finished,
	Error(std::io::Error),
}

struct ImportNotif {
	recv: Receiver<Progress>,
	latest: Progress,
	finished: Option<Instant>,
}

impl Notification for ImportNotif {
	fn ty(&self) -> NotifType {
		match self.latest {
			Progress::Error(_) => NotifType::Error,
			_ => NotifType::Info,
		}
	}

	fn draw(&mut self, ui: &mut Ui, _: &Fonts) {
		for progress in self.recv.try_iter() {
			self.latest = progress;
		}

		match &self.latest {
			Progress::Discovering => {
				ui.label("discovering");
			},
			Progress::Update(ratio) => {
				ui.add(ProgressBar::new(*ratio).show_percentage());
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
	OpenedSystem { sys: Arc<AssetSystem>, existed: bool },
}

fn import_thread(recv: Receiver<ThreadMsg>, send: Sender<ThreadRecv>) {
	tracy::set_thread_name(tracy::c_str!("importer"));
	let mut sys = None;
	for msg in recv {
		match msg {
			ThreadMsg::Open(path) => {
				let existed = path.exists();
				let Ok(s) = AssetSystem::new(&path) else {
					continue;
				};
				let s = Arc::new(s);
				let _ = send.send(ThreadRecv::OpenedSystem {
					sys: s.clone(),
					existed,
				});
				sys = Some(s);
			},
			ThreadMsg::Import(req) => {
				if let Some(ref sys) = sys {
					let importer = match GltfImporter::initialize(&req.from) {
						Some(Ok(x)) => x,
						Some(Err(e)) => {
							event!(Level::ERROR, "failed to import {:?}: {:?}", req.from, e);
							let _ = req.progress.send(Progress::Error(e));
							continue;
						},
						None => {
							let ext = req.from.extension().unwrap().to_str().unwrap();
							event!(Level::ERROR, "unsupported extension {}", ext);
							let _ = req.progress.send(Progress::Error(std::io::Error::other(format!(
								"unsupported extension {}",
								ext
							))));
							continue;
						},
					};
					if let Err(err) = importer.import(sys.view(req.to), |x| {
						event!(Level::INFO, "{:.2}%", x * 100.0);
						let _ = req.progress.send(Progress::Update(x));
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
	pub system: Option<Arc<AssetSystem>>,
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
		let mut this = Self {
			system: None,
			cursor: PathBuf::new(),
			send,
			recv: irecv,
		};
		if let Some(s) = std::env::args().nth(1) {
			this.open(s);
		}
		this
	}

	pub fn open(&mut self, path: impl Into<PathBuf>) {
		let buf = std::fs::canonicalize(path.into()).unwrap();
		self.send.send(ThreadMsg::Open(buf.clone())).unwrap();
		self.cursor = PathBuf::new();
	}

	pub fn render(
		&mut self, frame: &mut Frame, ctx: &Context, notifs: &mut NotifStack, renderer: &mut Renderer, fonts: &Fonts,
	) {
		for msg in self.recv.try_iter() {
			match msg {
				ThreadRecv::OpenedSystem { sys, existed } => {
					self.system = Some(sys);
					notifs.push(
						"project",
						PushNotif::new(
							NotifType::Info,
							format!("{} project", if existed { "loaded" } else { "created" }),
						),
					);
				},
			}
		}

		egui::TopBottomPanel::bottom("assets")
			.resizable(true)
			.min_height(100.0)
			.show(ctx, |ui| {
				if let Some(ref mut sys) = self.system {
					sys.tick(frame);
					let mut delete_target = None;
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
										.clicked()
									{
										self.cursor.pop();
									}
								});

								ui.separator();

								for (i, component) in self.cursor.components().enumerate() {
									if ui
										.text_button(RichText::new(component.as_os_str().to_string_lossy()).heading())
										.clicked()
									{
										self.cursor = self.cursor.components().take(i + 1).collect::<PathBuf>();
										break;
									}
									ui.label(RichText::new("/").heading());
								}
							});
							ui.add_space(5.0);

							let view = sys.view(self.cursor.clone());
							let count = view.entries().count();

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
										for (name, view) in view.entries().skip(start_obj).take(obj_range.len()) {
											match view {
												DirView::Dir => {
													grid.cell(|ui| {
														if ui
															.text_button(fonts.icons.text(icons::FOLDER).size(32.0))
															.double_clicked()
														{
															self.cursor.push(&*name);
														}
														ui.add(Label::new(&*name).truncate(true));
													});
												},
												DirView::Asset(h) => {
													grid.cell(|ui| {
														let button = ui.text_button(
															fonts
																.icons
																.text(if h.ty == Mesh::TYPE {
																	icons::CUBE
																} else if h.ty == Scene::TYPE {
																	icons::MAP
																} else {
																	icons::QUESTION
																})
																.size(32.0),
														);
														if button.double_clicked() {
															if h.ty == Scene::TYPE {
																renderer.set_scene(h.asset);
															}
														} else {
															button.context_menu(|ui| {
																if ui.button("delete").clicked() {
																	delete_target = Some(self.cursor.join(&name));
																}
															});
														}
														ui.add(Label::new(name).truncate(true));
													});
												},
											}
										}
									});
								});
						});

						if let Some(del) = delete_target {}
					}
				} else {
					ui.centered_and_justified(|ui| {
						ui.label(RichText::new("no project loaded").size(20.0));
					});
				}
			});
	}
}
