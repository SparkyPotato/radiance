use std::{
    ffi::c_void,
    mem,
    sync::atomic::{AtomicU32, Ordering},
};

use slang_sys::Interface;

use crate::{sys, utils};

trait BlobImpl {
    fn ref_count(&self) -> &AtomicU32;
    fn get_bytes(&self) -> &[u8];
}

unsafe extern "C" fn query_interface(
    this: *mut sys::ISlangUnknown,
    uuid: *const sys::SlangUUID,
    out_object: *mut *mut c_void,
) -> sys::SlangResult {
    if out_object.is_null() {
        return utils::E_INVALIDARG;
    }

    if libc::memcmp(
        uuid.cast(),
        &sys::slang_IBlob::UUID as *const _ as *const _,
        mem::size_of::<sys::SlangUUID>(),
    ) == 0
        || libc::memcmp(
            uuid.cast(),
            &utils::UNKNOWN_UUID as *const _ as *const _,
            mem::size_of::<sys::SlangUUID>(),
        ) == 0
    {
        ((*(*this).vtable_).ISlangUnknown_addRef)(this);
        *out_object = this.cast();
        utils::S_OK
    } else {
        utils::E_NOINTERFACE
    }
}

unsafe extern "C" fn add_ref<I: BlobImpl>(this: *mut sys::ISlangUnknown) -> u32 {
    (*this.cast::<I>())
        .ref_count()
        .fetch_add(1, Ordering::SeqCst)
}

unsafe extern "C" fn release<I: BlobImpl>(this: *mut sys::ISlangUnknown) -> u32 {
    let ref_count = (*this.cast::<I>())
        .ref_count()
        .fetch_sub(1, Ordering::SeqCst);
    if ref_count == 1 {
        let _ = Box::from_raw(this.cast::<I>());
    }

    ref_count
}

unsafe extern "C" fn get_buffer_pointer<I: BlobImpl>(this: *mut sys::slang_IBlob) -> *const c_void {
    (*this.cast::<I>()).get_bytes().as_ptr().cast()
}

unsafe extern "C" fn get_buffer_size<I: BlobImpl>(this: *mut sys::slang_IBlob) -> usize {
    (*this.cast::<I>()).get_bytes().len()
}

#[repr(C)]
pub(crate) struct StaticBlobImpl {
    vtbl: *const sys::slang_IBlobVtbl,
    ref_count: AtomicU32,
    value: &'static [u8],
}

impl StaticBlobImpl {
    #[inline]
    pub(crate) fn new(value: &'static [u8]) -> Self {
        const VTBL: sys::slang_IBlobVtbl = sys::slang_IBlobVtbl {
            _base: sys::ISlangUnknown__bindgen_vtable {
                ISlangUnknown_queryInterface: query_interface,
                ISlangUnknown_addRef: add_ref::<StaticBlobImpl>,
                ISlangUnknown_release: release::<StaticBlobImpl>,
            },
            getBufferPointer: get_buffer_pointer::<StaticBlobImpl>,
            getBufferSize: get_buffer_size::<StaticBlobImpl>,
        };

        Self {
            vtbl: &VTBL,
            ref_count: AtomicU32::new(1),
            value,
        }
    }
}

impl BlobImpl for StaticBlobImpl {
    #[inline]
    fn ref_count(&self) -> &AtomicU32 {
        &self.ref_count
    }

    #[inline]
    fn get_bytes(&self) -> &[u8] {
        &self.value
    }
}

#[repr(C)]
pub(crate) struct OwnedBlobImpl {
    vtbl: *const sys::slang_IBlobVtbl,
    ref_count: AtomicU32,
    value: Vec<u8>,
}

impl OwnedBlobImpl {
    #[inline]
    pub(crate) fn new(value: Vec<u8>) -> Self {
        const VTBL: sys::slang_IBlobVtbl = sys::slang_IBlobVtbl {
            _base: sys::ISlangUnknown__bindgen_vtable {
                ISlangUnknown_queryInterface: query_interface,
                ISlangUnknown_addRef: add_ref::<OwnedBlobImpl>,
                ISlangUnknown_release: release::<OwnedBlobImpl>,
            },
            getBufferPointer: get_buffer_pointer::<OwnedBlobImpl>,
            getBufferSize: get_buffer_size::<OwnedBlobImpl>,
        };

        Self {
            vtbl: &VTBL,
            ref_count: AtomicU32::new(1),
            value,
        }
    }
}

impl BlobImpl for OwnedBlobImpl {
    #[inline]
    fn ref_count(&self) -> &AtomicU32 {
        &self.ref_count
    }

    #[inline]
    fn get_bytes(&self) -> &[u8] {
        &self.value
    }
}
