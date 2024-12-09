#![feature(proc_macro_expand)]

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, LitStr};

#[proc_macro_derive(RadComponent, attributes(uuid))]
pub fn component(input: TokenStream) -> TokenStream {
	let inp: proc_macro2::TokenStream = input.clone().into();
	let i = parse_macro_input!(input as DeriveInput);

	let uuid = i
		.attrs
		.iter()
		.find_map(|x| {
			if x.path().is_ident("uuid") {
				x.parse_args::<LitStr>().ok()
			} else {
				None
			}
		})
		.expect("no uuid attribute found");

	let name = i.ident;
	let (im, ty, wh) = i.generics.split_for_impl();

	let path: TokenStream = quote! { std::module_path!() }.into();
	let path: proc_macro2::TokenStream = path.expand_expr().unwrap().into();

	quote! {
		use rad_world::{ReflectRadComponent, Component, ReflectComponent};
		use rad_world::bevy_reflect as bevy_reflect;

		rad_world::bevy_reflect::impl_reflect! {
			#[reflect(RadComponent, Component)]
			#[reflect(no_field_bounds)]
			#[type_path = #path]
			#inp
		}

		impl #im Component for #name #ty #wh {
			const STORAGE_TYPE: rad_world::StorageType = rad_world::StorageType::Table;
		}

		impl #im RadComponent for #name #ty #wh {
			fn uuid() -> rad_world::Uuid
			where
				Self: Sized { rad_world::uuid!(#uuid) }

			fn uuid_dyn(&self) -> rad_world::Uuid { rad_world::uuid!(#uuid) }
		}
	}
	.into()
}
