use std::time::{Duration, Instant};

use egui::Ui;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::ui::{
	notif::{NotifType, Notification},
	Fonts,
};

pub struct TaskPool {
	pool: ThreadPool,
}

pub enum Task<T> {
	Pending(oneshot::Receiver<T>),
	Resolved(T),
}

impl<T> Task<T> {
	pub fn get(&mut self) -> Option<&mut T> {
		match self {
			Self::Pending(r) => {
				let recv = match r.try_recv() {
					Ok(x) => x,
					Err(oneshot::TryRecvError::Empty) => return None,
					Err(oneshot::TryRecvError::Disconnected) => panic!("task disconnected"),
				};
				*self = Self::Resolved(recv);
				let Self::Resolved(r) = self else { unreachable!() };
				Some(r)
			},
			Self::Resolved(r) => Some(r),
		}
	}

	pub fn join(self) -> T {
		match self {
			Self::Pending(r) => r.recv().unwrap(),
			Self::Resolved(r) => r,
		}
	}
}

pub trait Progress: 'static {
	fn draw(&mut self, ui: &mut Ui, fonts: &Fonts);
}

impl Progress for String {
	fn draw(&mut self, ui: &mut Ui, _: &Fonts) { ui.label(self.as_str()); }
}

impl Progress for &'static str {
	fn draw(&mut self, ui: &mut Ui, _: &Fonts) { ui.label(*self); }
}

pub struct TaskNotif<T, P> {
	prog: P,
	task: Task<T>,
	resolved: Option<Instant>,
}

impl<T: Notification, P: Progress> TaskNotif<T, P> {
	pub fn new(prog: P, task: Task<T>) -> Self {
		Self {
			prog,
			task,
			resolved: None,
		}
	}

	fn get(&mut self) -> Option<&mut T> {
		let r = self.task.get();
		if self.resolved.is_none() && r.is_some() {
			self.resolved = Some(Instant::now());
		}
		r
	}
}

impl<T: Notification, P: Progress> Notification for TaskNotif<T, P> {
	fn ty(&mut self) -> NotifType {
		match self.get() {
			Some(r) => r.ty(),
			None => NotifType::Info,
		}
	}

	fn draw(&mut self, ui: &mut Ui, fonts: &Fonts) {
		match self.get() {
			Some(r) => r.draw(ui, fonts),
			None => self.prog.draw(ui, fonts),
		}
	}

	fn expired(&mut self, _: Duration) -> bool {
		self.get();
		let dur = self.resolved;
		match self.get() {
			Some(r) => r.expired(dur.unwrap().elapsed()),
			None => false,
		}
	}

	fn dismissable(&mut self) -> bool {
		match self.get() {
			Some(r) => r.dismissable(),
			None => false,
		}
	}
}

impl TaskPool {
	pub fn new() -> Self {
		Self {
			pool: ThreadPoolBuilder::new().build().unwrap(),
		}
	}

	pub fn spawn<T: Send + 'static>(&self, f: impl Send + 'static + FnOnce() -> T) -> Task<T> {
		let (s, r) = oneshot::channel();
		self.pool.spawn(move || {
			let _ = s.send(f());
		});
		Task::Pending(r)
	}
}
