use std::{path::PathBuf, sync::Arc, time::Instant};

use crossbeam_channel::{Receiver, Sender};
use egui::{Align, Align2, Context, FontId, Label, Layout, PointerButton, ProgressBar, RichText, ScrollArea, Ui};
use egui_extras::Size;
use egui_grid::GridBuilder;
use radiance_asset::{gltf::GltfImporter, mesh::Mesh, scene::Scene, Asset, AssetSystem, DirView, Importer, Uuid};
use radiance_graph::graph::Frame;
use rustc_hash::FxHashSet;
use tracing::{event, Level};

use crate::ui::{
	notif::{NotifStack, NotifType, Notification, PushNotif},
	render::Renderer,
	widgets::{icons, IntoIcon, TextButton, UiExt},
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
	creating_dir: Option<(String, bool)>,
	selection: FxHashSet<String>,
	send: Sender<ThreadMsg>,
	recv: Receiver<ThreadRecv>,
}

impl AssetManager {
	const CELL_SIZE: f32 = 75.0;

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
			creating_dir: None,
			selection: FxHashSet::default(),
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
				if let Some(ref sys) = self.system {
					sys.tick(frame);
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
										.text_button(fonts.icons.text(icons::PLUS).heading())
										.on_hover_text("new dir")
										.clicked()
									{
										self.creating_dir = Some(("new folder".to_string(), true));
									}
								});

								ui.separator();

								ui.vertical(|ui| {
									ui.add_space(2.5);
									if ui
										.text_button(fonts.icons.text(icons::ARROW_UP).heading())
										.on_hover_text("go back")
										.clicked()
									{
										self.cursor.pop();
										self.selection.clear();
									}
								});

								ui.separator();

								for (i, component) in self.cursor.components().enumerate() {
									if ui
										.text_button(RichText::new(component.as_os_str().to_string_lossy()).heading())
										.clicked()
									{
										self.cursor = self.cursor.components().take(i + 1).collect::<PathBuf>();
										self.selection.clear();
										break;
									}
									ui.label(RichText::new("/").heading());
								}
							});
							ui.add_space(5.0);

							self.draw_assets(ui, renderer, fonts);
						});
					}
				} else {
					ui.centered_and_justified(|ui| {
						ui.label(RichText::new("no project loaded").size(20.0));
					});
				}
			});
	}

	fn draw_assets(&mut self, ui: &mut Ui, renderer: &mut Renderer, fonts: &Fonts) {
		let Some(ref sys) = self.system else { return };

		let view = sys.view(self.cursor.clone());
		let count = view.entries().count() + self.creating_dir.is_some() as usize;

		if count == 0 {
			return;
		}

		let rect = ui.available_rect_before_wrap();
		let width = rect.width();
		let height = rect.height();
		let cells_x = (width / Self::CELL_SIZE) as usize - 1;
		let rows = (count + cells_x - 1) / cells_x;

		let mut interact = false;
		let mut delete = false;
		let mut mv = None;
		ScrollArea::vertical()
			.min_scrolled_width(width)
			.min_scrolled_height(height)
			.auto_shrink([false, false])
			.stick_to_right(true)
			.show_rows(ui, Self::CELL_SIZE, rows, |ui, range| {
				let mut grid = GridBuilder::new().layout_standard(Layout::top_down(Align::Center));
				for _ in range.clone() {
					grid = grid.new_row(Size::remainder());
					grid = grid.cells(Size::exact(Self::CELL_SIZE), cells_x as _);
				}

				let start_obj = range.start * cells_x;
				let end_obj = range.end * cells_x;
				let obj_range = start_obj..end_obj;

				grid.show(ui, |mut grid| {
					for (name, view) in view.entries().skip(start_obj).take(obj_range.len()) {
						grid.cell(|ui| {
							let (i, d, m) = match view {
								DirView::Dir => Self::draw_asset(ui, name, None, fonts, &mut self.selection, |n, s| {
									self.cursor.push(n);
									s.clear();
								}),
								DirView::Asset(h) => {
									Self::draw_asset(ui, name, Some(h.ty), fonts, &mut self.selection, |_, _| {
										if h.ty == Scene::TYPE {
											renderer.set_scene(h.asset);
										}
									})
								},
							};
							interact |= i;
							delete |= d;
							if let Some(m) = m {
								mv = Some(m);
							}
						})
					}

					grid.cell(|ui| {
						if let Some((ref mut dir, ref mut focus)) = self.creating_dir {
							ui.text_button(fonts.icons.text(icons::FOLDER).size(32.0));
							let tex = ui.text_edit_singleline(dir);
							if *focus {
								*focus = false;
								ui.memory_mut(|x| x.request_focus(tex.id));
							}
							if tex.lost_focus() {
								let _ = view.create_dir(&dir);
								self.creating_dir = None;
							}
						}
					})
				});
			});

		if !interact && ui.input(|x| x.pointer.any_click()) {
			self.selection.clear();
		} else if delete {
			for x in self.selection.drain() {
				let _ = view.delete(&x);
			}
		} else if let Some((name, mv)) = mv {
			let _ = view.move_into(&name, mv.iter().map(|x| x.as_str()));
		}
	}

	fn draw_asset(
		ui: &mut Ui, name: String, ty: Option<Uuid>, fonts: &Fonts, selection: &mut FxHashSet<String>,
		open: impl FnOnce(&str, &mut FxHashSet<String>),
	) -> (bool, bool, Option<(String, Arc<FxHashSet<String>>)>) {
		let style = ui.style();
		let button = egui::Frame::none()
			.fill(if selection.contains(&name) {
				style.visuals.selection.bg_fill
			} else {
				style.visuals.widgets.noninteractive.bg_fill
			})
			.show(ui, |ui| {
				let button = ui.add(TextButton::new_draggable(
					fonts.icons.text(Self::get_icon(ty)).size(32.0),
				));
				ui.add(Label::new(&name).truncate(true));
				button
			})
			.inner;

		let mut delete = false;
		let mut interact = false;
		let mut mv = None;
		if button.double_clicked_by(PointerButton::Primary) {
			open(&name, selection);
			selection.clear();
			interact = true;
		} else if button.clicked_by(PointerButton::Primary) {
			if !ui.input(|x| x.modifiers.shift) {
				selection.clear();
			}
			selection.insert(name);
			interact = true;
		} else if button.dragged_by(PointerButton::Primary) {
			if button.drag_started() {
				if !selection.contains(&name) {
					selection.clear();
				}
				selection.insert(name);
				button.dnd_set_drag_payload(selection.clone());
			}

			let count = selection.len();
			let rect = ui.ctx().debug_painter().text(
				ui.input(|x| x.pointer.latest_pos().unwrap()),
				Align2([Align::Center, Align::Center]),
				icons::FILE,
				FontId {
					size: 32.0,
					family: fonts.icons.clone().into(),
				},
				ui.style().visuals.widgets.inactive.fg_stroke.color,
			);
			if count > 1 {
				ui.ctx().debug_painter().text(
					rect.max,
					Align2([Align::Min, Align::Min]),
					format!("{count}"),
					FontId {
						size: 8.0,
						family: fonts.icons.clone().into(),
					},
					ui.style().visuals.selection.bg_fill,
				);
			}
		} else if let Some(moved) = button.dnd_release_payload::<FxHashSet<String>>() {
			mv = Some((name, moved));
		} else {
			if button.clicked_by(PointerButton::Secondary) {
				selection.insert(name);
			}
			button.context_menu(|ui| {
				interact = true;
				if ui.button("delete").clicked() {
					delete = true;
					ui.close_menu();
				}
			});
		}
		(interact, delete, mv)
	}

	fn get_icon(ty: Option<Uuid>) -> &'static str {
		match ty {
			Some(Mesh::TYPE) => icons::CUBE,
			Some(Scene::TYPE) => icons::MAP,
			Some(_) => icons::QUESTION,
			None => icons::FOLDER,
		}
	}
}
