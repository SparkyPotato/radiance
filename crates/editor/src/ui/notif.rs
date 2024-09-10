use std::{
	error::Error,
	time::{Duration, Instant},
};

use egui::{Align2, Area, Context, Frame, Id, Order, Resize, Ui};

use crate::ui::{
	widgets::{icons, IntoIcon},
	Fonts,
};

pub struct NotifStack {
	notifs: Vec<Notif>,
	dismissed: Option<usize>,
}

#[derive(Copy, Clone)]
pub enum NotifType {
	Info,
	Warning,
	Error,
}

pub trait Notification: 'static {
	fn ty(&mut self) -> NotifType;

	fn draw(&mut self, ui: &mut Ui, fonts: &Fonts);

	fn expired(&mut self, dur: Duration) -> bool;

	fn dismissable(&mut self) -> bool;
}

pub struct PushNotif {
	ty: NotifType,
	contents: String,
	life: Duration,
}

impl PushNotif {
	pub fn new(ty: NotifType, contents: impl ToString) -> Self {
		Self {
			ty,
			contents: contents.to_string(),
			life: Duration::from_secs(5),
		}
	}

	pub fn with_life(ty: NotifType, contents: impl ToString, life: Duration) -> Self {
		Self {
			ty,
			contents: contents.to_string(),
			life,
		}
	}
}

impl Notification for PushNotif {
	fn ty(&mut self) -> NotifType { self.ty }

	fn draw(&mut self, ui: &mut Ui, _: &Fonts) { ui.label(self.contents.clone()); }

	fn expired(&mut self, dur: Duration) -> bool { dur > self.life }

	fn dismissable(&mut self) -> bool { true }
}

impl<E: Error + 'static> Notification for Result<(), E> {
	fn ty(&mut self) -> NotifType {
		match self {
			Ok(()) => NotifType::Info,
			Err(_) => NotifType::Error,
		}
	}

	fn draw(&mut self, ui: &mut Ui, _: &Fonts) {
		ui.label(match self {
			Ok(()) => format!("success"),
			Err(e) => format!("failed: {e}"),
		});
	}

	fn expired(&mut self, dur: Duration) -> bool { dur > Duration::from_secs(2) && self.is_ok() }

	fn dismissable(&mut self) -> bool { true }
}

struct Notif {
	pub header: String,
	pub contents: Box<dyn Notification>,
	pub start: Instant,
}

impl NotifStack {
	pub fn new() -> Self {
		Self {
			notifs: Vec::new(),
			dismissed: None,
		}
	}

	pub fn push<T: Notification>(&mut self, header: impl ToString, notif: T) {
		self.notifs.push(Notif {
			header: header.to_string(),
			contents: Box::new(notif),
			start: Instant::now(),
		});
	}

	pub fn render(&mut self, ctx: &Context, fonts: &Fonts) {
		let rect = ctx.input(|x| x.screen_rect);
		let x = rect.width() - 15.0;
		let mut y = rect.height() - 15.0;
		let now = Instant::now();

		let mut i = 0;
		self.notifs.retain_mut(|x| {
			let ret = !x.contents.expired(now - x.start) && self.dismissed != Some(i);
			i += 1;
			ret
		});
		self.dismissed = None;

		for (i, notif) in self.notifs.iter_mut().enumerate().rev() {
			let resp = Area::new(Id::new("notifications").with(i))
				.pivot(Align2::RIGHT_BOTTOM)
				.fixed_pos((x, y))
				.interactable(true)
				.order(Order::Foreground)
				.show(ctx, |ui| {
					Frame::window(&*ctx.style()).show(ui, |ui| {
						Resize::default()
							.resizable(false)
							.default_width(150.0)
							.min_width(150.0)
							.default_height(50.0)
							.show(ui, |ui| {
								ui.horizontal(|ui| {
									match notif.contents.ty() {
										NotifType::Info => ui.heading(fonts.icons.text(icons::INFO)),
										NotifType::Warning => ui.heading(fonts.icons.text(icons::WARNING)),
										NotifType::Error => ui.heading(fonts.icons.text(icons::ERROR)),
									};

									ui.vertical(|ui| {
										ui.heading(&notif.header);
										notif.contents.draw(ui, fonts);
									});
								});
							});
					});
				});
			if resp.response.clicked() && notif.contents.dismissable() {
				self.dismissed = Some(i)
			}
			y -= resp.response.rect.height() + 5.0;
		}
	}
}
