use egui::{epaint, Color32, Label, Response, Sense, Stroke, Ui, Widget, WidgetInfo, WidgetType};

use crate::icons;

pub trait UiIconButton {
	fn icon_button(&mut self, icon: &str) -> Response;
}

impl UiIconButton for Ui {
	fn icon_button(&mut self, icon: &str) -> Response { self.add(TextButton::new(icon)) }
}

pub struct TextButton {
	label: Label,
}

impl TextButton {
	pub fn new(icon: &str) -> Self {
		Self {
			label: Label::new(icons::icon(icon)).sense(Sense::click()),
		}
	}

	pub fn new_draggable(icon: &str) -> Self {
		Self {
			label: Label::new(icons::icon(icon)).sense(Sense::click_and_drag()),
		}
	}
}

impl Widget for TextButton {
	fn ui(self, ui: &mut Ui) -> Response {
		let (pos, text_galley, response) = self.label.layout_in_ui(ui);
		response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, text_galley.text()));

		if ui.is_rect_visible(response.rect) {
			let response_color = if response.hovered() {
				ui.style().visuals.strong_text_color()
			} else {
				ui.style().visuals.weak_text_color()
			};

			ui.painter().add(epaint::TextShape {
				pos,
				galley: text_galley,
				override_text_color: Some(response_color),
				underline: Stroke::NONE,
				angle: 0.0,
				fallback_color: Color32::default(),
				opacity_factor: 1.0,
			});
		}

		response
	}
}
