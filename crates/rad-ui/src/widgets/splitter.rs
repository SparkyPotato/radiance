// https://gist.github.com/mkalte666/f9a982be0ac0276080d3434ab9ea4655

use std::hash::Hash;

use egui::{CursorIcon, Id, Pos2, Rect, Rounding, Sense, Ui, UiBuilder, Vec2};

/// An axis that a Splitter can use
#[derive(Copy, Clone, Debug)]
pub enum SplitterAxis {
	Horizontal,
	Vertical,
}

/// The internal data used by a splitter. Stored into memory
#[derive(Debug, Clone)]
struct SplitterData {
	axis: SplitterAxis,
	pos: f32,
	min_size: f32,
}

/// Splits a ui in half, using a draggable separator in the middle.
pub struct Splitter {
	id: Id,
	data: SplitterData,
}
impl Splitter {
	/// Create a new Splitter
	pub fn new(id_source: impl Hash, axis: SplitterAxis) -> Self {
		Self {
			id: Id::new(id_source),
			data: SplitterData {
				axis,
				pos: 0.5,
				min_size: 0.0,
			},
		}
	}

	/// Sets the minimum allowed size for the area
	pub fn min_size(mut self, points: f32) -> Self {
		self.data.min_size = points;
		self
	}

	/// Thes the default position of the splitter separator. Usually it sits in the center, this moves it around.
	pub fn default_pos(mut self, pos: f32) -> Self {
		self.data.pos = pos;
		self
	}

	pub fn show(self, ui: &mut Ui, add_contents: impl FnOnce(&mut Ui, &mut Ui)) {
		let mut data = if let Some(d) = ui.memory(|mem| mem.data.get_temp(self.id)) {
			d
		} else {
			self.data.clone()
		};

		let sep_size = 10.0;
		let sep_stroke = 2.0;
		let whole_area = ui.available_size();

		let split_axis_size = match data.axis {
			SplitterAxis::Horizontal => whole_area.x,
			SplitterAxis::Vertical => whole_area.y,
		};
		let split_a_size = (split_axis_size - sep_size) * data.pos;
		let split_b_size = split_axis_size - sep_size - split_a_size;

		let child_size_a = match data.axis {
			SplitterAxis::Horizontal => Vec2::new(split_a_size, whole_area.y),
			SplitterAxis::Vertical => Vec2::new(whole_area.x, split_a_size),
		};

		let child_size_b = match data.axis {
			SplitterAxis::Horizontal => Vec2::new(split_b_size, whole_area.y),
			SplitterAxis::Vertical => Vec2::new(whole_area.x, split_b_size),
		};

		let child_rect_a = Rect::from_min_size(ui.next_widget_position(), child_size_a);
		let mut ui_a = ui.new_child(UiBuilder::new().max_rect(child_rect_a));

		let sep_rect = match data.axis {
			SplitterAxis::Horizontal => Rect::from_min_size(
				Pos2::new(child_rect_a.max.x, child_rect_a.min.y),
				Vec2::new(sep_size, whole_area.y),
			),
			SplitterAxis::Vertical => Rect::from_min_size(
				Pos2::new(child_rect_a.min.x, child_rect_a.max.y),
				Vec2::new(whole_area.x, sep_size),
			),
		};

		let resp = ui.allocate_rect(sep_rect, Sense::hover().union(Sense::click_and_drag()));

		let sep_draw_rect = match data.axis {
			SplitterAxis::Horizontal => Rect::from_min_size(
				Pos2::new(sep_rect.min.x + sep_size / 2.0 - sep_stroke / 2.0, sep_rect.min.y),
				Vec2::new(sep_stroke, sep_rect.height()),
			),
			SplitterAxis::Vertical => Rect::from_min_size(
				Pos2::new(sep_rect.min.x, sep_rect.min.y + sep_size / 2.0 - sep_stroke / 2.0),
				Vec2::new(sep_rect.width(), sep_stroke),
			),
		};
		ui.painter().rect_filled(
			sep_draw_rect,
			Rounding::ZERO,
			ui.style().visuals.noninteractive().bg_stroke.color,
		);

		let child_rect_b = match data.axis {
			SplitterAxis::Horizontal => Rect::from_min_size(Pos2::new(sep_rect.max.x, sep_rect.min.y), child_size_b),
			SplitterAxis::Vertical => Rect::from_min_size(Pos2::new(sep_rect.min.x, sep_rect.max.y), child_size_b),
		};
		let mut ui_b = ui.new_child(UiBuilder::new().max_rect(child_rect_b));

		add_contents(&mut ui_a, &mut ui_b);

		if resp.hovered() {
			match data.axis {
				SplitterAxis::Horizontal => ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal),
				SplitterAxis::Vertical => ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical),
			}
		}

		if resp.dragged() {
			let delta_pos = match data.axis {
				SplitterAxis::Horizontal => resp.drag_delta().x / whole_area.x,
				SplitterAxis::Vertical => resp.drag_delta().y / whole_area.y,
			};

			data.pos += delta_pos;
		}

		// clip pos
		let min_pos = (data.min_size / split_axis_size).min(1.0);
		let max_pos = (1.0 - min_pos).max(0.0);
		data.pos = data.pos.max(min_pos).min(max_pos);

		ui.memory_mut(|mem| {
			mem.data.insert_temp(self.id, data);
		})
	}
}
