use std::sync::{Arc, LazyLock};

use egui::{Context, FontData, FontDefinitions, FontFamily};

const INTER: &[u8] = include_bytes!("../fonts/Inter/Inter-Regular.otf");
const FONT_AWESOME: &[u8] = include_bytes!("../fonts/Font Awesome/Font Awesome 6 Free-Solid-900.otf");

pub static ICONS: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("Icon"));

pub fn setup_fonts(ctx: &Context) {
	let mut fonts = FontDefinitions::default();
	fonts
		.font_data
		.insert("Inter".to_string(), FontData::from_static(INTER));
	fonts
		.font_data
		.insert("Font Awesome".to_string(), FontData::from_static(FONT_AWESOME));
	fonts
		.families
		.insert(FontFamily::Proportional, vec!["Inter".to_string()]);
	fonts
		.families
		.insert(FontFamily::Name(ICONS.clone()), vec!["Font Awesome".to_string()]);
	ctx.set_fonts(fonts);
}
