use std::fmt::Debug;

use thiserror::Error;

use crate::{sys, Blob};

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Result(sys::SlangResult),
    #[error("{:?}", .0.as_str().unwrap_or(""))]
    Blob(Blob),
}

impl Default for Error {
    #[inline]
    fn default() -> Self {
        Self::Result(E_INVALIDARG)
    }
}

pub trait IntoError<T> {
    fn into_error(self) -> Result<T>;
}

pub type Result<T> = std::result::Result<T, Error>;

impl<T> IntoError<T> for Option<T> {
    #[inline]
    fn into_error(self) -> Result<T> {
        self.ok_or(Error::default())
    }
}

#[inline]
pub(crate) fn result_from_ffi(result: sys::SlangResult) -> Result<()> {
    if result < 0 {
        Err(Error::Result(result))
    } else {
        Ok(())
    }
}

#[inline]
pub(crate) fn result_from_blob(
    result: sys::SlangResult,
    blob: *mut sys::slang_IBlob,
) -> Result<()> {
    if result < 0 {
        Err(Error::Blob(Blob(blob)))
    } else {
        Ok(())
    }
}

macro_rules! define_interface {
    ($name: ident, $sys_ty: ty) => {
        paste::paste! {
            #[repr(transparent)]
            pub struct $name(*mut $sys_ty);

            impl $name {
                #[inline]
                pub fn as_raw(&self) -> *mut $sys_ty {
                    self.0
                }
            }

            impl Clone for $name {
                fn clone(&self) -> Self {
                    unsafe {
                        ((*self.0).unknown_vtable().ISlangUnknown_addRef)(self.0.cast());
                    }
                    Self(self.0.cast())
                }
            }
        }
    };
    ($name: ident, $sys_ty: ty, Debug) => {
        paste::paste! {
            #[repr(transparent)]
            #[derive(Debug)]
            pub struct $name(*mut $sys_ty);

            impl $name {
                #[inline]
                pub fn as_raw(&self) -> *mut $sys_ty {
                    self.0
                }
            }

            impl Clone for $name {
                fn clone(&self) -> Self {
                    unsafe {
                        ((*self.0).unknown_vtable().ISlangUnknown_addRef)(self.0.cast());
                    }
                    Self(self.0.cast())
                }
            }
        }

        //TODO: ref types
    };

    ($name: ident, $sys_ty: ty, $base_ty: ty) => {
        paste::paste! {
            #[repr(transparent)]
            #[derive(Debug)]
            pub struct $name(*mut $sys_ty);

            impl $name {
                #[inline]
                pub fn as_raw(&self) -> *mut $sys_ty {
                    self.0
                }
            }

            impl Deref for $name {
                type Target = $base_ty;

                #[inline]
                fn deref(&self) -> &Self::Target {
                    unsafe { mem::transmute(self) }
                }
            }

            impl DerefMut for $name {
                #[inline]
                fn deref_mut(&mut self) -> &mut Self::Target {
                    unsafe { mem::transmute(self) }
                }
            }
        }
    };
}

pub(crate) use define_interface;

macro_rules! assert_size_and_align {
    ($type_: ty, $ffi_type: ty) => {
        ::static_assertions::assert_eq_size!($type_, $ffi_type);
        ::static_assertions::assert_eq_align!($type_, $ffi_type);
    };
}

pub(crate) use assert_size_and_align;

pub(crate) const UNKNOWN_UUID: sys::SlangUUID = sys::SlangUUID {
    data1: 0x67618701,
    data2: 0xd116,
    data3: 0x468f,
    data4: [0xab, 0x3b, 0x47, 0x4b, 0xed, 0xce, 0xe, 0x3d],
};

pub(crate) const S_OK: sys::SlangResult = 0;
pub(crate) const E_INVALIDARG: sys::SlangResult = -2147024809;
pub(crate) const E_NOINTERFACE: sys::SlangResult = -2147467262;
