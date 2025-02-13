use std::{
    ffi::{c_char, c_void, CStr},
    mem, ptr,
    sync::atomic::{AtomicU32, Ordering},
};

use slang_sys::Interface;

use crate::{sys, utils, Error, FileSystem};

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
            &sys::ISlangCastable::UUID as *const _ as *const _,
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

unsafe extern "C" fn add_ref(this: *mut sys::ISlangUnknown) -> u32 {
    (*this.cast::<FileSystemImpl>())
        .ref_count
        .fetch_add(1, Ordering::SeqCst)
}

unsafe extern "C" fn release(this: *mut sys::ISlangUnknown) -> u32 {
    let ref_count = (*this.cast::<FileSystemImpl>())
        .ref_count
        .fetch_sub(1, Ordering::SeqCst);
    if ref_count == 1 {
        let _ = Box::from_raw(this.cast::<FileSystemImpl>());
    }

    ref_count
}

unsafe extern "C" fn cast_as(this: *mut sys::ISlangCastable, guid: &sys::SlangUUID) -> *mut c_void {
    let mut object = ptr::null_mut();
    if query_interface(this.cast(), guid as *const _, &mut object) == utils::S_OK {
        object
    } else {
        ptr::null_mut()
    }
}

unsafe extern "C" fn load_file(
    this: *mut sys::ISlangFileSystem,
    path: *const c_char,
    out_blob: *mut *mut sys::slang_IBlob,
) -> sys::SlangResult {
    if out_blob.is_null() {
        return utils::E_INVALIDARG;
    }

    let wrapper = &mut (*this.cast::<FileSystemImpl>()).wrapper;

    let path = CStr::from_ptr(path).to_string_lossy();

    match wrapper.load_file(&path) {
        Ok(blob) => {
            *out_blob = blob.0;
            mem::forget(blob);
            utils::S_OK
        }
        Err(error) => {
            match error {
                Error::Result(result) => result,
                Error::Blob(blob) => {
                    *out_blob = blob.0;
                    mem::forget(blob);
                    utils::E_INVALIDARG
                }
            }
        }
    }
}

const VTBL: sys::ISlangFileSystemVtbl = sys::ISlangFileSystemVtbl {
    _base: sys::ISlangCastableVtbl {
        _base: sys::ISlangUnknown__bindgen_vtable {
            ISlangUnknown_queryInterface: query_interface,
            ISlangUnknown_addRef: add_ref,
            ISlangUnknown_release: release,
        },
        castAs: cast_as,
    },
    loadFile: load_file,
};

#[repr(C)]
pub(crate) struct FileSystemImpl {
    vtbl: *const sys::ISlangFileSystemVtbl,
    ref_count: AtomicU32,
    wrapper: Box<dyn FileSystem>,
}

impl FileSystemImpl {
    #[inline]
    pub(crate) fn new(wrapper: Box<dyn FileSystem>) -> Self {
        Self {
            vtbl: &VTBL,
            ref_count: AtomicU32::new(1),
            wrapper,
        }
    }
}
