use egui::{FontFamily, RichText};

use crate::fonts::ICONS;

pub const ARROW_UP: &str = "\u{f062}";

pub const BRUSH: &str = "\u{f55d}";
pub const FILE: &str = "\u{f15b}";
pub const FOLDER: &str = "\u{f07b}";
pub const MAP: &str = "\u{f279}";
pub const IMAGE: &str = "\u{f03e}";
pub const CUBE: &str = "\u{f1b2}";
pub const QUESTION: &str = "\u{3f}";

pub const INFO: &str = "\u{f05a}";
pub const WARNING: &str = "\u{f071}";
pub const ERROR: &str = "\u{f06a}";

pub const PLUS: &str = "\u{2b}";

pub fn icon(icon: &str) -> RichText { RichText::from(icon).family(FontFamily::Name(ICONS.clone())) }
