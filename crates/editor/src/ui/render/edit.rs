use std::{collections::VecDeque, time::Duration};

use egui::{Label, Sense, SidePanel, Ui};
use egui_extras::{Column, TableBuilder};
use radiance_asset::{
	rref::{RRef, RWeak},
	scene::Scene,
};

use crate::ui::{
	notif::{NotifStack, NotifType, PushNotif},
	render::picking::Picker,
	widgets::splitter::{Splitter, SplitterAxis},
};

pub struct Editor {
	changes: VecDeque<Change>,
	last_save: usize,
	last_scene: Option<RWeak>,
}

enum Change {
	Rename { node: u32, from: String },
}

impl Editor {
	pub fn new() -> Self {
		Self {
			changes: VecDeque::new(),
			last_save: 0,
			last_scene: None,
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
				let h = top.available_height();
				TableBuilder::new(top)
					.sense(Sense::click())
					.column(Column::remainder())
					.auto_shrink([false; 2])
					.striped(true)
					.min_scrolled_height(h)
					.max_scroll_height(h)
					.body(|body| {
						body.rows(18.0, scene.instance_count() as _, |mut row| {
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

				if let Some(sel) = picker.get_sel() {
					let name = &mut *scene.node_name(sel);
					let resp = bot.text_edit_singleline(name);
					if resp.gained_focus() {
						self.changes.push_back(Change::Rename {
							node: sel,
							from: name.clone(),
						});
					} else if resp.lost_focus() {
						match self.changes.pop_back() {
							Some(Change::Rename { node, from })
								if node == sel && (from == *name || name.is_empty()) =>
							{
								*name = from;
							},
							x => self.changes.extend(x),
						}
					}
				} else {
					bot.centered_and_justified(|ui| {
						ui.label("no selection");
					});
				}
			})
		});
	}

	pub fn is_dirty(&mut self, scene: &RRef<Scene>) -> bool {
		self.check_scene(scene);
		self.changes.is_empty() || self.last_save == self.changes.len() - 1
	}

	pub fn undo(&mut self, scene: &RRef<Scene>, notifs: &mut NotifStack) {
		self.check_scene(scene);
		if let Some(c) = self.changes.pop_back() {
			match c {
				Change::Rename { node, from } => {
					let mut n = scene.node_name(node);
					notifs.push(
						"undo",
						PushNotif::with_life(
							NotifType::Info,
							format!("rename '{}' to '{}'", from, *n),
							Duration::from_secs(2),
						),
					);
					*n = from;
				},
			}
		}
	}
}
