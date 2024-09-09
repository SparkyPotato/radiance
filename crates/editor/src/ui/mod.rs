//! The entire editor UI.

mod assets;
mod debug;
mod notif;
mod render;
mod widgets;

use egui::{menu, Context, Key, KeyboardShortcut, Modifiers, TopBottomPanel};
use radiance_graph::{device::Device, graph::Frame, Result};
use rfd::FileDialog;
pub use widgets::Fonts;
use winit::event::WindowEvent;

use crate::{
	ui::{assets::AssetManager, debug::Debug, notif::NotifStack, render::Renderer},
	window::Window,
};

pub struct UiState {
	fonts: Fonts,
	assets: AssetManager,
	debug: Debug,
	renderer: Renderer,
	notifs: NotifStack,
}

impl UiState {
	pub fn new(device: &Device, fonts: Fonts) -> Result<Self> {
		Ok(Self {
			fonts,
			debug: Debug::new(),
			assets: AssetManager::new(),
			renderer: Renderer::new(device)?,
			notifs: NotifStack::new(),
		})
	}

	pub fn render<'pass>(
		&'pass mut self, frame: &mut Frame<'pass, '_>, ctx: &Context, window: &Window, arena_size: usize,
	) {
		let mut undo = ctx.input_mut(|x| {
			x.consume_shortcut(&KeyboardShortcut {
				modifiers: Modifiers::COMMAND,
				logical_key: Key::Z,
			})
		});
		let mut save = ctx.input_mut(|x| {
			x.consume_shortcut(&KeyboardShortcut {
				modifiers: Modifiers::COMMAND,
				logical_key: Key::S,
			})
		});

		TopBottomPanel::top("menu").show(ctx, |ui| {
			menu::bar(ui, |ui| {
				ui.menu_button("file", |ui| {
					let new = ui.button("new").clicked();
					let load = ui.button("load").clicked();
					if new || load {
						if let Some(path) = FileDialog::new().pick_folder() {
							self.assets.open(path);
						}
					}
					if ui.button("save").clicked() {
						save = true;
					}
				});

				ui.menu_button("edit", |ui| {
					if ui.button("undo").clicked() {
						undo = true;
					}
				});

				ui.menu_button("window", |ui| {
					ui.checkbox(&mut self.debug.enabled, "debug");
				});

				ui.menu_button("cameras", |ui| self.renderer.draw_camera_menu(ui));
			});
		});

		if undo {
			self.renderer.undo(&mut self.notifs);
		}
		if save {
			self.renderer.save(self.assets.system.as_deref(), &mut self.notifs);
		}

		self.debug.set_arena_size(arena_size);
		self.debug.render(frame.device(), ctx);

		self.assets
			.render(frame, ctx, &mut self.notifs, &mut self.renderer, &self.fonts);
		self.renderer.render(
			frame,
			ctx,
			window,
			&self.debug,
			self.assets.system.as_deref().map(|x| &*x),
		);

		self.notifs.render(ctx, &self.fonts);
	}

	pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
		self.renderer.on_window_event(window, event);
	}

	pub unsafe fn destroy(self, device: &Device) { self.renderer.destroy(device); }
}
