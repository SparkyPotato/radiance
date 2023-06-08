use std::sync::Arc;

use egui::{
	epaint,
	FontData,
	FontDefinitions,
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

const INTER: &[u8] = include_bytes!("../../../fonts/Inter/Inter-Regular.otf");
const FONT_AWESOME: &[u8] = include_bytes!("../../../fonts/Font Awesome/Font Awesome 6 Free-Solid-900.otf");

#[derive(Clone)]
pub struct Fonts {
	pub icons: Font,
}

impl Fonts {
	pub fn defs() -> (FontDefinitions, Fonts) {
		let mut fonts = FontDefinitions::empty();
		fonts
			.font_data
			.insert("Inter".to_string(), FontData::from_static(INTER));
		fonts
			.font_data
			.insert("Font Awesome".to_string(), FontData::from_static(FONT_AWESOME));
		let icon: Arc<str> = Arc::from("Icon");
		fonts
			.families
			.insert(FontFamily::Proportional, vec!["Inter".to_string()]);
		fonts
			.families
			.insert(FontFamily::Name(icon.clone()), vec!["Font Awesome".to_string()]);

		let this = Self { icons: Font(icon) };

		(fonts, this)
	}
}

#[derive(Clone)]
pub struct Font(Arc<str>);

impl Into<FontFamily> for Font {
	fn into(self) -> FontFamily { FontFamily::Name(self.0) }
}

pub trait IntoIcon {
	fn text(&self, icon: impl Into<RichText>) -> RichText;
}

impl IntoIcon for Font {
	fn text(&self, icon: impl Into<RichText>) -> RichText { icon.into().family(self.clone().into()) }
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
