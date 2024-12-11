use rad_core::asset::aref::ARef;
use rad_renderer::assets::image::Image;
use rad_ui::{
	egui::{self, Context, Id, Sense, Window},
	raw_texture_to_id,
};

pub struct ImagePreviewer {
	previews: Vec<ARef<Image>>,
}

impl ImagePreviewer {
	pub fn new() -> Self { Self { previews: Vec::new() } }

	pub fn render(&mut self, ctx: &Context) {
		let mut yeet = Vec::new();
		for (i, img) in self.previews.iter().enumerate() {
			let mut open = true;
			Window::new("image preview")
				.id(Id::new("image preview").with(img.asset_id()))
				.open(&mut open)
				.show(ctx, |ui| {
					let rect = ui.available_rect_before_wrap();
					let mut size = rect.size();
					ui.allocate_rect(rect, Sense::focusable_noninteractive());
					let desc = img.image().desc();
					let aspect = desc.size.width as f32 / desc.size.height as f32;
					if size.x / aspect < size.y {
						size.y = size.x / aspect;
					} else {
						size.x = size.y * aspect;
					}
					ui.put(
						rect,
						egui::Image::new((raw_texture_to_id(img.view().id.unwrap()), size)),
					);
				});

			if !open {
				yeet.push(i);
			}
		}

		for i in yeet.into_iter().rev() {
			self.previews.swap_remove(i);
		}
	}

	pub fn add_preview(&mut self, image: ARef<Image>) { self.previews.push(image); }
}
