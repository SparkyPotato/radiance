use egui::{Align2, Context, Id, Window};

use crate::ui::{
	widgets::{icons, IntoIcon},
	Fonts,
};

pub struct NotifStack {
	push: Vec<Notif>,
	editable: Vec<Notif>,
	order: Vec<u32>,
}

pub enum NotifType {
	Info,
	Warning,
	Error,
}

pub enum NotifContents {
	Simple { header: String, body: String },
}

pub struct Notif {
	pub ty: NotifType,
	pub contents: NotifContents,
}

pub struct NotifId(u32);

impl NotifStack {
	pub fn new() -> Self {
		Self {
			push: Vec::new(),
			editable: Vec::new(),
			order: Vec::new(),
		}
	}

	pub fn push(&mut self, notif: Notif) {
		let id = self.push.len();
		self.push.push(notif);
		self.order.push(id as u32);
	}

	pub fn editable(&mut self, notif: Notif) -> NotifId {
		let id = self.editable.len();
		self.editable.push(notif);
		self.order.push((id as u32) | 1 << 31);
		NotifId(id as _)
	}

	pub fn edit(&mut self, id: NotifId, notif: Notif) { self.editable[id.0 as usize] = notif; }

	pub fn render(&mut self, ctx: &Context, fonts: &Fonts) {
		let Self { push, editable, order } = self;

		let rect = ctx.input(|x| x.screen_rect);
		let x = rect.width() - 15.0;
		let mut y = rect.height() - 15.0;
		for &id in order.iter() {
			let resp = Window::new("")
				.id(Id::new("notifications").with(id))
				.title_bar(false)
				.resizable(false)
				.open(&mut true)
				.pivot(Align2::RIGHT_BOTTOM)
				.fixed_pos((x, y))
				.show(ctx, |ui| {
					let notif = if id & (1 << 31) == 0 {
						&push[id as usize]
					} else {
						&editable[id as usize & !(1 << 31)]
					};

					ui.horizontal(|ui| {
						match notif.ty {
							NotifType::Info => ui.heading(fonts.icons.text(icons::INFO)),
							NotifType::Warning => ui.heading(fonts.icons.text(icons::WARNING)),
							NotifType::Error => ui.heading(fonts.icons.text(icons::ERROR)),
						};

						ui.vertical(|ui| match &notif.contents {
							NotifContents::Simple { header, body } => {
								ui.heading(header);
								ui.label(body);
							},
						});
					});
				})
				.unwrap();
			y -= resp.response.rect.height() + 5.0;
		}
	}
}
