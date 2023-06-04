use std::sync::Arc;

use egui::{
	epaint,
	FontFamily,
	Label,
	Response,
	RichText,
	Sense,
	Stroke,
	Ui,
	Widget,
	WidgetInfo,
	WidgetText,
	WidgetType,
};

pub mod icons;

pub trait IntoIcon {
	fn text(&self, icon: impl Into<RichText>) -> RichText;
}

impl IntoIcon for Arc<str> {
	fn text(&self, icon: impl Into<RichText>) -> RichText { icon.into().family(FontFamily::Name(self.clone())) }
}

pub trait UiExt {
	fn icon_button(&mut self, icon: &impl IntoIcon, text: impl Into<RichText>) -> Response;

	fn text_button(&mut self, text: impl Into<WidgetText>) -> Response;
}

impl UiExt for Ui {
	fn icon_button(&mut self, icon: &impl IntoIcon, text: impl Into<RichText>) -> Response {
		self.add(TextButton::new(icon.text(text)))
	}

	fn text_button(&mut self, text: impl Into<WidgetText>) -> Response { self.add(TextButton::new(text)) }
}

pub struct TextButton {
	label: Label,
}

impl TextButton {
	pub fn new(text: impl Into<WidgetText>) -> Self {
		Self {
			label: Label::new(text).sense(Sense::click()),
		}
	}
}

impl Widget for TextButton {
	fn ui(self, ui: &mut Ui) -> Response {
		let (pos, text_galley, response) = self.label.layout_in_ui(ui);
		response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, text_galley.text()));

		if ui.is_rect_visible(response.rect) {
			let response_color = if response.hovered() {
				ui.style().visuals.strong_text_color()
			} else {
				ui.style().visuals.weak_text_color()
			};

			ui.painter().add(epaint::TextShape {
				pos,
				galley: text_galley.galley,
				override_text_color: Some(response_color),
				underline: Stroke::NONE,
				angle: 0.0,
			});
		}

		response
	}
}
