use std::{collections::VecDeque, sync::Arc, time::Duration};

use egui::{DragValue, Label, Rect, Sense, SidePanel, Ui};
use egui_extras::{Column, TableBuilder};
use radiance_asset::{
	rref::{RRef, RWeak},
	scene::{Scene, SceneReader, Transform},
	AssetSystem,
};
use radiance_graph::graph::Frame;
use transform_gizmo_egui::{Gizmo, GizmoConfig, GizmoExt, GizmoMode, GizmoOrientation, GizmoVisuals};
use vek::{Quaternion, Vec3, Vec4};

use crate::ui::{
	notif::{NotifStack, NotifType, PushNotif},
	render::{camera::CameraController, picking::Picker},
	task::{TaskNotif, TaskPool},
	widgets::splitter::{Splitter, SplitterAxis},
};

pub struct Editor {
	changes: VecDeque<Change>,
	last_save: usize,
	last_scene: Option<RWeak>,
	gizmo: Gizmo,
	gizmo_moving: bool,
}

enum Change {
	Rename { node: u32, from: String },
	Transform { node: u32, from: Transform },
}

impl Editor {
	pub fn new() -> Self {
		let mut gizmo = Gizmo::default();
		let c = *gizmo.config();
		gizmo.update_config(GizmoConfig {
			modes: GizmoMode::all_translate(),
			visuals: GizmoVisuals {
				gizmo_size: 150.0,
				..Default::default()
			},
			..c
		});
		Self {
			changes: VecDeque::new(),
			last_save: 0,
			last_scene: None,
			gizmo,
			gizmo_moving: false,
		}
	}

	fn check_scene(&mut self, scene: &RRef<Scene>) {
		let weak = Some(scene.downgrade());
		if weak != self.last_scene {
			self.changes.clear();
			self.last_scene = weak;
			self.last_save = 0;
		}
	}

	pub fn render(&mut self, ui: &mut Ui, scene: &RRef<Scene>, picker: &mut Picker) {
		self.check_scene(scene);
		SidePanel::right("scene").show_inside(ui, |ui| {
			Splitter::new("scene split", SplitterAxis::Vertical).show(ui, |top, bot| {
				top.push_id("scene top", |ui| {
					let h = ui.available_height();
					TableBuilder::new(ui)
						.sense(Sense::click())
						.column(Column::remainder())
						.auto_shrink([false; 2])
						.striped(true)
						.min_scrolled_height(h)
						.max_scroll_height(h)
						.body(|body| {
							body.rows(18.0, scene.node_count() as _, |mut row| {
								let i = row.index() as u32;
								row.set_selected(picker.get_sel() == Some(i));
								if row
									.col(|ui| {
										ui.add(Label::new(&scene.node(i).name).truncate(true));
									})
									.1
									.clicked()
								{
									picker.select(i);
								}
							})
						});
				});

				bot.push_id("scene bot", |ui| {
					if let Some(sel) = picker.get_sel() {
						let mut node = scene.edit_node(sel);

						let resp = ui.text_edit_singleline(&mut node.name);
						if resp.gained_focus() {
							self.changes.push_back(Change::Rename {
								node: sel,
								from: node.name.clone(),
							});
						} else if resp.lost_focus() {
							match self.changes.pop_back() {
								Some(Change::Rename { node: o, from })
									if o == sel && (from == node.name || node.name.is_empty()) =>
								{
									node.name = from;
								},
								x => self.changes.extend(x),
							}
						}

						let t = node.transform;
						let speed = 0.005;
						let dec = 2;
						TableBuilder::new(ui)
							.columns(Column::auto(), 4)
							.auto_shrink([false; 2])
							.body(|mut body| {
								let value = |this: &mut Self, ui: &mut Ui, val: &mut f32| {
									let resp = ui.add(DragValue::new(val).fixed_decimals(dec).speed(speed));
									if resp.gained_focus() || (!resp.has_focus() && resp.drag_started()) {
										this.changes.push_back(Change::Transform { node: sel, from: t });
									}
								};

								body.row(18.0, |mut row| {
									row.col(|ui| {
										ui.label("location");
									});
									row.col(|ui| value(self, ui, &mut node.transform.translation.x));
									row.col(|ui| value(self, ui, &mut node.transform.translation.y));
									row.col(|ui| value(self, ui, &mut node.transform.translation.z));
								});
								body.row(18.0, |mut row| {
									let mut eul = quat_to_eul(node.transform.rotation);
									let old = eul;
									row.col(|ui| {
										ui.label("rotation");
									});
									row.col(|ui| value(self, ui, &mut eul.x));
									row.col(|ui| value(self, ui, &mut eul.y));
									row.col(|ui| value(self, ui, &mut eul.z));
									if old != eul {
										node.transform.rotation = eul_to_quat(eul);
									}
								});
								body.row(18.0, |mut row| {
									row.col(|ui| {
										ui.label("scale");
									});
									row.col(|ui| value(self, ui, &mut node.transform.scale.x));
									row.col(|ui| value(self, ui, &mut node.transform.scale.y));
									row.col(|ui| value(self, ui, &mut node.transform.scale.z));
								});
								body.row(18.0, |mut row| {
									row.col(|ui| {
										ui.label("gizmo");
									});
									let c = *self.gizmo.config();
									row.col(|ui| {
										let t = GizmoMode::all_translate();
										if ui.selectable_label(c.modes == t, "translate").clicked() {
											self.gizmo.update_config(GizmoConfig { modes: t, ..c });
										}
									});
									row.col(|ui| {
										let r = GizmoMode::all_rotate();
										if ui.selectable_label(c.modes == r, "rotate").clicked() {
											self.gizmo.update_config(GizmoConfig { modes: r, ..c });
										}
									});
									row.col(|ui| {
										let s = GizmoMode::all_scale();
										if ui.selectable_label(c.modes == s, "scale").clicked() {
											self.gizmo.update_config(GizmoConfig { modes: s, ..c });
										}
									});
								});
							});
					} else {
						ui.centered_and_justified(|ui| {
							ui.label("no selection");
						});
					}
				});
			})
		});
	}

	pub fn draw_gizmo<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, ui: &mut Ui, rect: Rect, scene: &'pass RRef<Scene>,
		cam: &CameraController, picker: &Picker,
	) -> SceneReader {
		if let Some(sel) = picker.get_sel() {
			let mut node = scene.edit_node(sel);
			let t = to_gizmo_transform(node.transform);
			let c = self.gizmo.config();
			let cam = cam.get();
			self.gizmo.update_config(GizmoConfig {
				view_matrix: cam.view.as_().into(),
				projection_matrix: cam.projection(rect.aspect_ratio()).as_().into(),
				viewport: rect,
				orientation: GizmoOrientation::Global,
				..*c
			});
			let ret = self.gizmo.interact(ui, &[t]);
			let last_focus = self.gizmo_moving;
			let new_focus = ret.is_some();
			self.gizmo_moving = new_focus;
			if !last_focus && new_focus {
				self.changes.push_back(Change::Transform {
					node: sel,
					from: from_gizmo_transform(t),
				});
			}
			if let Some((_, t)) = ret {
				node.transform = from_gizmo_transform(t[0]);
			}
		}

		scene.tick(frame)
	}

	pub fn is_dirty(&mut self, scene: &RRef<Scene>) -> bool {
		self.check_scene(scene);
		self.last_save != self.changes.len()
	}

	pub fn save(
		&mut self, system: Option<Arc<AssetSystem>>, scene: &RRef<Scene>, notifs: &mut NotifStack, pool: &TaskPool,
	) {
		match system {
			Some(sys) if self.is_dirty(scene) => {
				let s = scene.clone();
				notifs.push("save scene", TaskNotif::new("saving", pool.spawn(move || sys.save(&s))));
				self.last_save = self.changes.len();
			},
			_ => {},
		}
	}

	pub fn undo(&mut self, scene: &RRef<Scene>, notifs: &mut NotifStack) {
		self.check_scene(scene);
		if let Some(c) = self.changes.pop_back() {
			match c {
				Change::Rename { node, from } => {
					let mut n = scene.edit_node(node);
					notifs.push(
						"undo",
						PushNotif::with_life(
							NotifType::Info,
							format!("rename '{}' to '{}'", from, n.name),
							Duration::from_secs(2),
						),
					);
					n.name = from;
				},
				Change::Transform { node, from } => {
					let mut n = scene.edit_node(node);
					notifs.push(
						"undo",
						PushNotif::with_life(
							NotifType::Info,
							format!("transform '{}'", n.name),
							Duration::from_secs(2),
						),
					);
					n.transform = from;
				},
			}
		}
	}
}

fn to_gizmo_transform(t: Transform) -> transform_gizmo_egui::math::Transform {
	transform_gizmo_egui::math::Transform::from_scale_rotation_translation(
		t.scale.as_(),
		Quaternion::from(t.rotation.into_vec4().as_()),
		t.translation.as_(),
	)
}

fn from_gizmo_transform(t: transform_gizmo_egui::math::Transform) -> Transform {
	Transform {
		translation: Vec3::<f64>::from(t.translation).as_(),
		rotation: Vec4::<f64>::from(Quaternion::from(t.rotation)).as_().into(),
		scale: Vec3::<f64>::from(t.scale).as_(),
	}
}

fn quat_to_eul(q: Quaternion<f32>) -> Vec3<f32> {
	Vec3::new(
		(2.0 * (q.x * q.w + q.y * q.z)).atan2(1.0 - 2.0 * (q.x * q.x + q.y * q.y)),
		(2.0 * (q.y * q.w - q.x * q.z)).asin(),
		(2.0 * (q.z * q.w + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
	)
	.map(|x| x.to_degrees())
}

fn eul_to_quat(v: Vec3<f32>) -> Quaternion<f32> {
	let v = v.map(|x| x.to_radians());
	Quaternion::identity().rotated_x(v.x).rotated_y(v.y).rotated_z(v.z)
}
