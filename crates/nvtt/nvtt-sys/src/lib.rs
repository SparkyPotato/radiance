#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl From<NvttBoolean> for bool {
    fn from(val: NvttBoolean) -> Self {
        match val {
            NvttBoolean::NVTT_True => true,
            NvttBoolean::NVTT_False => false,
        }
    }
}

impl From<bool> for NvttBoolean {
    fn from(other: bool) -> Self {
        match other {
            true => NvttBoolean::NVTT_True,
            false => NvttBoolean::NVTT_False,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        unsafe {
            assert_eq!(30106, crate::nvttVersion());
        }
    }

    #[test]
    fn cuda_disabled() {
        unsafe {
            let context = crate::nvttCreateContext();
            crate::nvttSetContextCudaAcceleration(context, false.into());
            let cuda_enabled: bool = crate::nvttContextIsCudaAccelerationEnabled(context).into();
            assert!(!cuda_enabled);

            crate::nvttDestroyContext(context);
        }
    }

    #[test]
    fn cuda_enabled() {
        unsafe {
            let context = crate::nvttCreateContext();
            let cuda_supported: bool = crate::nvttIsCudaSupported().into();

            // Only passes on CUDA capable GPUs
            crate::nvttSetContextCudaAcceleration(context, cuda_supported.into());
            // Enable when possible
            let cuda_enabled: bool = crate::nvttContextIsCudaAccelerationEnabled(context).into();
            assert_eq!(cuda_enabled, cuda_supported);

            crate::nvttDestroyContext(context);
        }
    }
}
