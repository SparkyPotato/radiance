use crate::{ISlangUnknown__bindgen_vtable, SlangUUID};

pub unsafe trait Interface : Sized {
    const UUID: SlangUUID;

    type VTable;

    #[inline]
    unsafe fn vtable(&mut self) -> &Self::VTable {
        &**(self as *mut Self as *mut *mut _)
    }

    #[inline]
    unsafe fn unknown_vtable(&mut self) -> &ISlangUnknown__bindgen_vtable {
        &**(self as *mut Self as *mut *mut _)
    }

}

pub use paste::paste;

macro_rules! interface {
    ($name: ident, [$data1: literal, $data2: literal, $data3: literal, {$data41: literal, $data42: literal, $data43: literal, $data44: literal, $data45: literal, $data46: literal, $data47: literal, $data48: literal}], {
        $($fn_name: ident: $fn_ty: ty,)*
    }) => {
        $crate::paste! {
            unsafe impl $crate::Interface for $name {
                const UUID: SlangUUID = SlangUUID { data1: $data1, data2: $data2, data3: $data3, data4: [$data41, $data42, $data43, $data44, $data45, $data46, $data47, $data48] };

                type VTable = [<$name Vtbl>];
            }

            #[repr(C)]
            pub struct [<$name Vtbl>] {
                pub _base: ISlangUnknown__bindgen_vtable,

                $(pub $fn_name: $fn_ty,)*
            }
        }
    };

    ($name: ident, [$data1: literal, $data2: literal, $data3: literal, {$data41: literal, $data42: literal, $data43: literal, $data44: literal, $data45: literal, $data46: literal, $data47: literal, $data48: literal}]: $base: ident, {
        $($fn_name: ident: $fn_ty: ty,)*
    }) => {
        $crate::paste! {
            unsafe impl $crate::Interface for $name {
                const UUID: SlangUUID = SlangUUID { data1: $data1, data2: $data2, data3: $data3, data4: [$data41, $data42, $data43, $data44, $data45, $data46, $data47, $data48] };

                type VTable = [<$name Vtbl>];
            }

            #[repr(C)]
            pub struct [<$name Vtbl>] {
                pub _base: [<$base Vtbl>],

                $(pub $fn_name: $fn_ty,)*
            }
        }
    };
}

pub(crate) use interface;

#[macro_export]
macro_rules! vtable_call {
	($ptr: expr, $method: ident($($args: expr),*)) => {
		((*$ptr).vtable().$method)($ptr.cast(), $($args),*)
	};
}