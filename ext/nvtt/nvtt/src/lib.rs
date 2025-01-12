//! # nvtt
//!
//! A rust wrapper around the [Nvidia Texture Tools 3 library](https://developer.nvidia.com/gpu-accelerated-texture-compression).
//!
//! NVTT 3 is a library that can be used to compress image data and files into compressed texture formats,
//! and to handle compressed and uncompressed images.
//!
//! In NVTT 3, most compression algorithms and image processing algorithms can be accelerated by the GPU.
//! These have CPU fallbacks for GPUs without support for CUDA. CUDA operations can be enabled
//! through the `cuda` feature.
//!
//! # Dependencies
//!
//! The NVTT 3 SDK must be installed on the system. A non-standard path to the binaries can be specified
//! via the `NVTT_PATH` environment variable. A compiler supporting at least C99 and dynamic linking is
//! also required.
//!
//! ## Windows
//!
//! Windows 10 or 11 (64-bit) are required. The `Path` environment variable must also contain
//! the path to the directory containing `nvtt.dll`. Note that this must be done manually, it
//! is not done in a standard Nvidia Texture Tools install.
//!
//! ## Linux
//!
//! 64-bit only; Ubuntu 16.04+ or a similarly compatible distro is required. `libc.so` version 6 or
//! higher is required as well.
//!
//! # Limitations
//!
//! Currently there is no file I/O support, no low-level (`nvtt_lowlevel.h`) wrapper, and no batch compression.
//!
//! # Using nvtt
//!
//! ```
//! # use nvtt_rs::{Context, CompressionOptions, CUDA_SUPPORTED, Format, InputFormat, OutputOptions, Surface};
//! // Create a surface
//! let input = InputFormat::Bgra8Ub {
//!     data: &[0u8; 16 * 16 * 4],
//!     unsigned_to_signed:  false,
//! };
//! let image = Surface::image(input, 16, 16, 1).unwrap();
//!
//! // Create the compression context; enable CUDA if possible
//! let mut context = Context::new();
//! #[cfg(feature = "cuda")]
//! if *CUDA_SUPPORTED {
//!     context.set_cuda_acceleration(true);
//! }
//!
//! // Specify compression settings to use; compress to Bc7
//! let mut compression_options = CompressionOptions::new();
//! compression_options.set_format(Format::Bc7);
//!
//! // Specify how to write the compressed data; indicate as sRGB
//! let mut output_options = OutputOptions::new();
//! output_options.set_srgb_flag(true);
//!
//! // Write the DDS header.
//! let header = context.output_header(
//!     &image,
//!     1, // number of mipmaps
//!     &compression_options,
//!     &output_options,
//! ).unwrap();
//!
//! // Compress and write the compressed data.
//! let bytes = context.compress(
//!     &image,
//!     &compression_options,
//!     &output_options,
//! ).unwrap();
//!
//! // Bc7 is 1 byte per pixel.
//! assert_eq!(16 * 16, bytes.len());
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

mod enums;
pub use enums::*;

use lazy_static::lazy_static;

lazy_static! {
    /// Whether CUDA acceleration is supported by this device.
    pub static ref CUDA_SUPPORTED: bool = {
        unsafe {
            // According to the docs, nvttUseCurrentDevice() must be called before any CUDA
            // operations are attempted.
            nvtt_sys::nvttUseCurrentDevice();
            nvtt_sys::nvttIsCudaSupported().into()
        }
    };
}

/// Return NVTT version.
pub fn version() -> u32 {
    unsafe { nvtt_sys::nvttVersion() }
}

use nvtt_sys::NvttCompressionOptions;
/// Describes the desired compression format and other compression settings.
pub struct CompressionOptions(*mut NvttCompressionOptions);

impl CompressionOptions {
    /// Constructs a new `CompressionOptions` struct. Sets compression options to the default values.
    pub fn new() -> Self {
        unsafe {
            let ptr = nvtt_sys::nvttCreateCompressionOptions();
            if ptr.is_null() {
                panic!("failed to allocate");
            } else {
                Self(ptr)
            }
        }
    }

    /// Sets compression options to the default values.
    pub fn reset(&mut self) {
        unsafe { nvtt_sys::nvttResetCompressionOptions(self.0) }
    }

    /// Set desired compression format.
    pub fn set_format(&mut self, format: Format) {
        unsafe { nvtt_sys::nvttSetCompressionOptionsFormat(self.0, format.into()) }
    }

    /// Set compression quality settings.
    pub fn set_quality(&mut self, quality: Quality) {
        unsafe { nvtt_sys::nvttSetCompressionOptionsQuality(self.0, quality.into()) }
    }

    /// Set the weights of each color channel used to measure compression error.
    ///
    /// The choice for these values is subjective. In most cases uniform color weights `(1.0, 1.0, 1.0)`
    /// work very well. A popular choice is to use the NTSC luma encoding weights `(0.2126, 0.7152, 0.0722)`,
    /// but I think that blue contributes to our perception more than 7%. A better choice in my opinion is `(3, 4, 2)`.
    ///
    /// # Optional Parameters
    /// - `alpha`: Defaults to `1.0`
    pub fn set_color_weights(&mut self, red: f32, green: f32, blue: f32, alpha: Option<f32>) {
        let alpha = alpha.unwrap_or(1.0);
        unsafe { nvtt_sys::nvttSetCompressionOptionsColorWeights(self.0, red, green, blue, alpha) }
    }

    /// Describes an RGB/RGBA format using 32-bit masks per channel.
    ///
    /// Note that this sets the number of bits per channel to 0.
    ///
    /// # Safety
    ///
    /// See Nvidia SDK documentation for more information.
    pub unsafe fn set_pixel_format(
        &mut self,
        bitcount: u32,
        rmask: u32,
        gmask: u32,
        bmask: u32,
        amask: u32,
    ) {
        unsafe {
            nvtt_sys::nvttSetCompressionOptionsPixelFormat(
                self.0, bitcount, rmask, gmask, bmask, amask,
            )
        }
    }

    /// Set pixel type.
    ///
    /// These are used for [`Format::Rgb`]: they indicate how the output should be interpreted,
    /// but do not have any influence over the input. They are ignored for other compression modes.
    pub fn set_pixel_type(&mut self, pixel_type: PixelType) {
        unsafe { nvtt_sys::nvttSetCompressionOptionsPixelType(self.0, pixel_type.into()) }
    }

    /// Set pitch alignment in bytes.
    pub fn set_pitch_alignment(&mut self, pitch_alignment: i32) {
        unsafe { nvtt_sys::nvttSetCompressionOptionsPitchAlignment(self.0, pitch_alignment) }
    }

    /// Set quantization options.
    ///
    /// # Optional Parameters
    /// - `alpha_threshold`: Defaults to `127`
    ///
    /// # Warning
    ///
    /// Do not enable dithering unless you know what you are doing. Quantization introduces errors.
    /// It's better to let the compressor quantize the result to minimize the error, instead of
    /// quantizing the data before handling it to the compressor.
    pub fn set_quantization(
        &mut self,
        color_dithering: bool,
        alpha_dithering: bool,
        binary_alpha: bool,
        alpha_threshold: Option<i32>,
    ) {
        let alpha_threshold = alpha_threshold.unwrap_or(127);
        unsafe {
            nvtt_sys::nvttSetCompressionOptionsQuantization(
                self.0,
                color_dithering.into(),
                alpha_dithering.into(),
                binary_alpha.into(),
                alpha_threshold,
            )
        }
    }

    /// Translate to and from D3D formats.
    ///
    /// Returns 0 if no corresponding format could be found.
    ///
    /// For [`Format::Rgb`], this looks at the pixel type and pixel format to determine
    /// the corresponding D3D format. For BC6, BC7, and ASTC, this returns a FourCC code: 'BC6H'
    /// for both unsigned and signed BC6, 'BC7L' for BC7, and 'ASTC' for all ASTC formats.
    pub fn d3d9_format(&self) -> u32 {
        unsafe { nvtt_sys::nvttGetCompressionOptionsD3D9Format(self.0) }
    }
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CompressionOptions {
    fn drop(&mut self) {
        unsafe {
            nvtt_sys::nvttDestroyCompressionOptions(self.0);
        }
    }
}

macro_rules! make_thread_local {
    ($buffer:ident) => {
        std::thread_local! {
            static $buffer: core::cell::RefCell<Vec<u8>> = core::cell::RefCell::new(Vec::new());
        }

        extern "C" fn output_callback(
            data_ptr: *const libc::c_void,
            len: libc::c_int,
        ) -> nvtt_sys::NvttBoolean {
            let len = len as usize;
            let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, len) };
            $buffer.with(|b| b.borrow_mut().extend_from_slice(data));
            true.into()
        }

        // Clear in case it was used before
        $buffer.with(|b| b.borrow_mut().clear());
    };
}

macro_rules! write_output {
        ($buffer:ident, $func:ident, $output_options:expr, $($arg:expr),* $(,)?) => {
        unsafe {
            nvtt_sys::nvttSetOutputOptionsOutputHandler($output_options.0, None, Some(output_callback), None);
            let res: bool = $func($($arg,)*).into();
            res.then_some($buffer.with(|b| b.replace(Vec::new())))
        }
        };
    }

use nvtt_sys::NvttContext;
/// Compression context.
pub struct Context(*mut NvttContext);

impl Context {
    /// Constructs a new compression context.
    pub fn new() -> Self {
        unsafe {
            // Creating a Context seems to default with CUDA enabled. So, this must be called
            // beforehand.
            nvtt_sys::nvttUseCurrentDevice();
            let ptr = nvtt_sys::nvttCreateContext();
            if ptr.is_null() {
                panic!("failed to allocate");
            } else {
                // Disable it by default, in case the cuda feature is not enabled.
                nvtt_sys::nvttSetContextCudaAcceleration(ptr, false.into());
                Self(ptr)
            }
        }
    }

    /// Enable/Disable CUDA acceleration; initializes CUDA if not already initialized.
    ///
    /// # Panics
    ///
    /// Panics if [`struct@CUDA_SUPPORTED`] is false.
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    #[cfg(feature = "cuda")]
    pub fn set_cuda_acceleration(&mut self, enable: bool) {
        if *CUDA_SUPPORTED {
            unsafe { nvtt_sys::nvttSetContextCudaAcceleration(self.0, enable.into()) }
        } else {
            panic!("cuda is not supported");
        }
    }

    /// Check if CUDA acceleration is enabled.
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    #[cfg(feature = "cuda")]
    pub fn is_cuda_acceleration_enabled(&self) -> bool {
        if !*CUDA_SUPPORTED {
            false
        } else {
            unsafe { nvtt_sys::nvttContextIsCudaAccelerationEnabled(self.0).into() }
        }
    }

    #[must_use]
    /// Write the [`Container`]'s header to a `Vec<u8>`. Returns `Some(Vec<u8>)` on success.
    pub fn output_header(
        &self,
        img: &Surface,
        mipmap_count: u32,
        compression_options: &CompressionOptions,
        output_options: &OutputOptions,
    ) -> Option<Vec<u8>> {
        let func = nvtt_sys::nvttContextOutputHeader;
        make_thread_local!(BUFFER);
        write_output!(
            BUFFER,
            func,
            output_options,
            self.0,
            img.0,
            mipmap_count as i32,
            compression_options.0,
            output_options.0,
        )
    }

    #[must_use]
    /// Write the [`Container`]'s header to a `Vec<u8>`. Returns `Some(Vec<u8>)` on success.
    pub fn output_header_cube(
        &self,
        cube: &CubeSurface,
        mipmap_count: u32,
        compression_options: &CompressionOptions,
        output_options: &OutputOptions,
    ) -> Option<Vec<u8>> {
        let func = nvtt_sys::nvttContextOutputHeaderCube;
        make_thread_local!(CUBE_HEADER_BUFFER);
        write_output!(
            CUBE_HEADER_BUFFER,
            func,
            output_options,
            self.0,
            cube.0,
            mipmap_count as i32,
            compression_options.0,
            output_options.0,
        )
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    /// Write the [`Container`]'s header to a `Vec<u8>`. Returns `Some(Vec<u8>)` on success.
    pub fn output_header_data(
        &self,
        tex_type: TextureType,
        w: u32,
        h: u32,
        d: u32,
        mipmap_count: u32,
        is_normal_map: bool,
        compression_options: &CompressionOptions,
        output_options: &OutputOptions,
    ) -> Option<Vec<u8>> {
        let func = nvtt_sys::nvttContextOutputHeaderData;
        make_thread_local!(DATA_HEADER_BUFFER);
        write_output!(
            DATA_HEADER_BUFFER,
            func,
            output_options,
            self.0,
            tex_type.into(),
            w as i32,
            h as i32,
            d as i32,
            mipmap_count as i32,
            is_normal_map.into(),
            compression_options.0,
            output_options.0,
        )
    }

    /// Compress the [`Surface`] and write the compressed data to a `Vec<u8>`. Returns `Some(Vec<u8>)` on success.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nvtt_rs::{Context, CompressionOptions, CUDA_SUPPORTED, Format, InputFormat, OutputOptions, Surface};
    /// let input = InputFormat::Bgra8Ub {
    ///     data: &[0u8; 100 * 100 * 4],
    ///     unsigned_to_signed:  false,
    /// };
    /// let image = Surface::image(input, 100, 100, 1).unwrap();
    /// let context = Context::new();
    ///
    /// let mut compression_options = CompressionOptions::new();
    /// // Just leave it as is
    /// compression_options.set_format(Format::Rgba);
    ///
    /// let output_options = OutputOptions::new();
    ///
    /// // Compress and write the compressed data.
    /// let bytes = context.compress(
    ///     &image,
    ///     &compression_options,
    ///     &output_options,
    /// ).unwrap();
    ///
    /// assert_eq!(100 * 100 * 4, bytes.len());
    /// for byte in bytes {
    ///     assert_eq!(0, byte);
    /// }
    /// ```
    pub fn compress(
        &self,
        img: &Surface,
        compression_options: &CompressionOptions,
        output_options: &OutputOptions,
    ) -> Option<Vec<u8>> {
        // Ignored
        let face = 0;
        let mipmap = 0;

        let func = nvtt_sys::nvttContextCompress;
        make_thread_local!(BUFFER);
        write_output!(
            BUFFER,
            func,
            output_options,
            self.0,
            img.0,
            face,
            mipmap,
            compression_options.0,
            output_options.0,
        )
    }

    /// Compress the [`CubeSurface`] and write the compressed data to a `Vec<u8>`. Returns `Some(Vec<u8>)` on success.
    pub fn compress_cube(
        &self,
        cube: &CubeSurface,
        compression_options: &CompressionOptions,
        output_options: &OutputOptions,
    ) -> Option<Vec<u8>> {
        // Ignored
        let mipmap = 0;

        let func = nvtt_sys::nvttContextCompressCube;
        make_thread_local!(CUBE_BUFFER);
        write_output!(
            CUBE_BUFFER,
            func,
            output_options,
            self.0,
            cube.0,
            mipmap,
            compression_options.0,
            output_options.0,
        )
    }

    /// Compress and write data to a `Vec<u8>`. Returns `Some(Vec<u8>)` on success.
    ///
    /// # Panics
    ///
    /// Panics if `w * h * d < rgba.len()`.
    pub fn compress_data(
        &self,
        w: u32,
        h: u32,
        d: u32,
        rgba: &[f32],
        compression_options: &CompressionOptions,
        output_options: &OutputOptions,
    ) -> Option<Vec<u8>> {
        if w * h * d < rgba.len() as u32 {
            panic!("rgba does match dimensions");
        }

        // Ignored
        let face = 0;
        let mipmap = 0;

        let func = nvtt_sys::nvttContextCompressData;
        make_thread_local!(DATA_BUFFER);
        write_output!(
            DATA_BUFFER,
            func,
            output_options,
            self.0,
            w as i32,
            h as i32,
            d as i32,
            face,
            mipmap,
            rgba.as_ptr(),
            compression_options.0,
            output_options.0,
        )
    }

    /// Returns the total compressed size of mips `0...mipmap_count - 1`, without compressing the image.
    ///
    /// Note that this does not include the container header, and mips are assumed to be tightly packed.
    ///
    /// For instance, call this with `mipmap_count` = [`Surface::count_mipmaps()`] and add the size of the DDS header
    /// to get the size of a DDS file with a surface and a full mip chain.
    pub fn estimate_size(
        &self,
        img: &Surface,
        mipmap_count: u32,
        compression_options: &CompressionOptions,
    ) -> u32 {
        unsafe {
            nvtt_sys::nvttContextEstimateSize(
                self.0,
                img.0,
                mipmap_count as i32,
                compression_options.0,
            ) as u32
        }
    }

    /// Returns the total compressed size of mips `0...mipmap_count - 1`, without compressing the image.
    ///
    /// Note that this does not include the container header, and mips are assumed to be tightly packed.
    ///
    /// For instance, call this with `mipmap_count` = [`CubeSurface::count_mipmaps()`] and add the size of the DDS header
    /// to get the size of a DDS file with a surface and a full mip chain.
    pub fn estimate_size_cube(
        &self,
        cube: &CubeSurface,
        mipmap_count: u32,
        compression_options: &CompressionOptions,
    ) -> u32 {
        unsafe {
            nvtt_sys::nvttContextEstimateSizeCube(
                self.0,
                cube.0,
                mipmap_count as i32,
                compression_options.0,
            ) as u32
        }
    }

    /// Returns the total compressed size of mips `0...mipmap_count - 1`, without compressing the image.
    ///
    /// Note that this does not include the container header, and mips are assumed to be tightly packed.
    pub fn estimate_size_data(
        &self,
        w: u32,
        h: u32,
        d: u32,
        mipmap_count: u32,
        compression_options: &CompressionOptions,
    ) -> u32 {
        unsafe {
            nvtt_sys::nvttContextEstimateSizeData(
                self.0,
                w as i32,
                h as i32,
                d as i32,
                mipmap_count as i32,
                compression_options.0,
            ) as u32
        }
    }

    // nvttContextQuantize is not in the windows dll, even though it is in the C header.
    // So this function is not usable at the moment
    //
    // /// Quantize a Surface to the number of bits per channel of the given format.
    // ///
    // /// This shouldn't be called unless you're confident you want to do this. Compressors quantize
    // /// automatically, and calling this will cause compression to minimize error with respect to
    // /// the quantized values, rather than the original image.
    // ///
    // /// See also [`Surface::quantize()`] and [`Surface::binarize()`].
    // ///
    // /// # Safety
    // ///
    // /// See Nvidia SDK documentation for more information.
    // pub unsafe fn quantize(&self, tex: &mut Surface, compression_options: &CompressionOptions) {
    //     unsafe { nvtt_sys::nvttContextQuantize(self.0, tex.0, compression_options.0) }
    // }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            nvtt_sys::nvttDestroyContext(self.0);
        }
    }
}

use nvtt_sys::NvttCubeSurface;
/// One level of a cube map texture.
///
/// Contained are six square [`Surface`]s numbered 0 through 5, all with the same size (referred to as the edge length).
/// By convention, these are the +x, -x, +y, -y, +z, and -z faces, in that order, of a cube map in a right-handed coordinate system.
pub struct CubeSurface(*mut NvttCubeSurface);

impl CubeSurface {
    /// Creates a [`CubeSurface`] from a 2D unfolded [`Surface`] in `img`.  
    ///
    /// # Panics
    ///
    /// Panics if `img` is 3D, or if it does not have a shape which can fold from the specified `layout`.
    /// See also [`CubeLayout::dim_supported()`].
    pub fn fold(img: &Surface, layout: CubeLayout) -> Self {
        if img.depth() > 1 {
            panic!("3D surface was provided");
        }

        if !layout.dim_supported(img.width(), img.height()) {
            panic!("layout does not support dimensions of img");
        }

        unsafe {
            let ptr = nvtt_sys::nvttCreateCubeSurface();
            if ptr.is_null() {
                panic!("failed to allocate");
            }

            nvtt_sys::nvttCubeSurfaceFold(ptr, img.0, layout.into());
            Self(ptr)
        }
    }

    /// Creates a [`Surface`] containing an unfolded/flattened representation of the cube surface.
    pub fn unfold(&self, layout: CubeLayout) -> Surface {
        unsafe {
            let ptr = nvtt_sys::nvttCubeSurfaceUnfold(self.0, layout.into());
            if ptr.is_null() {
                panic!("failed to allocate");
            }

            Surface(ptr)
        }
    }

    /// Returns a reference for the given face.
    pub fn face(&self, face: CubeFace) -> &Surface {
        unsafe { &*(nvtt_sys::nvttCubeSurfaceFace(self.0, face as i32) as *const Surface) }
    }

    /// Returns the edge length of any of the faces.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CubeSurface, CubeLayout, Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let bytes = [0u8; (10 * 60 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    ///
    /// let surface = Surface::image(input, 10, 60, 1).unwrap();
    /// let cube_surface = CubeSurface::fold(&surface, CubeLayout::Column);
    ///
    /// assert_eq!(10, cube_surface.edge_length());
    /// ```
    pub fn edge_length(&self) -> u32 {
        unsafe { nvtt_sys::nvttCubeSurfaceEdgeLength(self.0) as u32 }
    }

    /// Returns the number of mips that would be in a full mipmap chain starting with this [`CubeSurface`]
    ///
    /// For instance, a full mip chain for a cube map with 10x10 faces would consist of cube maps with sizes
    /// 10x10, 5x5, 2x2, and 1x1, and this function would return 4.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CubeSurface, CubeLayout, Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let bytes = [0u8; (10 * 60 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    ///
    /// let surface = Surface::image(input, 10, 60, 1).unwrap();
    /// let cube_surface = CubeSurface::fold(&surface, CubeLayout::Column);
    ///
    /// assert_eq!(4, cube_surface.count_mipmaps());
    /// ```
    pub fn count_mipmaps(&self) -> u32 {
        unsafe { nvtt_sys::nvttCubeSurfaceCountMipmaps(self.0) as u32 }
    }

    /// Computes an average value for the given channel over the entire sphere.
    ///
    /// This takes solid angles into account when producing an average per steradian,
    /// so texels near face edges are weighted less than texels near face edges.
    ///
    /// No gamma correction is performed, unlike [`Surface::average()`].
    pub fn average(&self, channel: Channel) -> f32 {
        unsafe { nvtt_sys::nvttCubeSurfaceAverage(self.0, channel as i32) }
    }

    /// Returns the minimum and maximum values respectively in the given channel.
    pub fn range(&self, channel: Channel) -> (f32, f32) {
        let mut min: f32 = 0.0;
        let mut max: f32 = 0.0;

        unsafe {
            nvtt_sys::nvttCubeSurfaceRange(
                self.0,
                channel as i32,
                &mut min as *mut _,
                &mut max as *mut _,
            );
            (min, max)
        }
    }

    /// Clamps values in the given channel to the range `[low, high]`.
    pub fn clamp(&mut self, channel: Channel, low: f32, high: f32) {
        unsafe {
            nvtt_sys::nvttCubeSurfaceClamp(self.0, channel as i32, low, high);
        }
    }

    /// Raises channels to the power `1/gamma`. `gamma=2.2` approximates sRGB-to-linear conversion.
    pub fn to_gamma(&mut self, gamma: f32) {
        unsafe { nvtt_sys::nvttCubeSurfaceToGamma(self.0, gamma) }
    }

    /// Raises channels to the power `gamma`. `gamma=2.2` approximates sRGB-to-linear conversion.
    pub fn from_gamma(&mut self, gamma: f32) {
        unsafe { nvtt_sys::nvttCubeSurfaceToLinear(self.0, gamma) }
    }

    /// Spherically convolves this [`CubeSurface`] with a `max(0.0f, cos(theta))^cosinePower` kernel,
    /// returning a [`CubeSurface`] with faces with dimension `size x size`.
    ///
    /// This is useful for generating prefiltered cube maps, as this corresponds to the cosine power
    /// used in the Phong reflection model (with energy conservation).
    ///
    /// This handles how each cube map texel can have a different solid angle. It also only considers
    /// texels for which the value of the kernel (without normalization) is at least 0.001.
    ///
    /// # Panics
    ///
    /// Panics is `size` is `0`.
    pub fn cosine_power_filter(
        &self,
        size: u32,
        cosine_power: f32,
        fixup_method: EdgeFixup,
    ) -> Self {
        if size == 0 {
            panic!("size cannot be zero");
        }

        unsafe {
            let ptr = nvtt_sys::nvttCubeSurfaceCosinePowerFilter(
                self.0,
                size as i32,
                cosine_power,
                fixup_method.into(),
            );
            if ptr.is_null() {
                panic!("failed to allocate");
            }

            Self(ptr)
        }
    }

    /// Produces a resized version of this [`CubeSurface`] using nearest-neighbor sampling.
    ///
    /// # Panics
    ///
    /// Panics is `size` is `0`.
    pub fn fast_resample(&self, size: u32, fixup_method: EdgeFixup) -> Self {
        unsafe {
            let ptr =
                nvtt_sys::nvttCubeSurfaceFastResample(self.0, size as i32, fixup_method.into());
            if ptr.is_null() {
                panic!("failed to allocate");
            }

            Self(ptr)
        }
    }
}

impl Drop for CubeSurface {
    fn drop(&mut self) {
        unsafe {
            nvtt_sys::nvttDestroyCubeSurface(self.0);
        }
    }
}

use nvtt_sys::NvttOutputOptions;
/// Holds container type and options specific to the container.
pub struct OutputOptions(*mut NvttOutputOptions);

impl OutputOptions {
    /// Constructs a new `OutputOptions` struct. Sets output options to the default values.
    pub fn new() -> Self {
        unsafe {
            let ptr = nvtt_sys::nvttCreateOutputOptions();
            if ptr.is_null() {
                panic!("failed to allocate");
            }

            Self(ptr)
        }
    }

    /// Set output header.
    pub fn set_output_header(&mut self, output_header: bool) {
        unsafe { nvtt_sys::nvttSetOutputOptionsOutputHeader(self.0, output_header.into()) }
    }

    /// Set container.
    pub fn set_container(&mut self, container: Container) {
        unsafe { nvtt_sys::nvttSetOutputOptionsContainer(self.0, container.into()) }
    }

    /// Set user version.
    pub fn set_user_version(&mut self, version: i32) {
        unsafe { nvtt_sys::nvttSetOutputOptionsUserVersion(self.0, version) }
    }

    /// Set the sRGB flag, indicating whether this file stores data with an sRGB transfer
    /// function (`true`) or a linear transfer function (`false`).
    pub fn set_srgb_flag(&mut self, b: bool) {
        unsafe { nvtt_sys::nvttSetOutputOptionsSrgbFlag(self.0, b.into()) }
    }
}

impl Default for OutputOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for OutputOptions {
    fn drop(&mut self) {
        unsafe {
            nvtt_sys::nvttDestroyOutputOptions(self.0);
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Bounding box of a [`Surface`]. For use with [`Surface::add_channel()`] and
/// [`Surface::create_sub_image()`].
pub struct BoundingBox {
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
    pub min_z: u32,
    pub max_z: u32,
}

impl BoundingBox {
    fn contains(self, other: Self) -> bool {
        #[rustfmt::skip]
        let ret = self.min_x <= other.min_x && other.max_x <= self.max_x &&
                  self.min_y <= other.min_y && other.max_y <= self.max_y &&
                  self.min_z <= other.min_z && other.max_z <= self.max_z;
        ret
    }

    fn same_shape(self, other: Self) -> bool {
        #[rustfmt::skip]
        let ret = (self.max_x - self.min_x) == (other.max_x - other.min_x) &&
                  (self.max_y - self.min_y) == (other.max_y - other.min_y) &&
                  (self.max_z - self.min_z) == (other.max_z - other.min_z);
        ret
    }
}

use nvtt_sys::NvttSurface;
/// One level of a 2D or 3D texture with red, green, blue, and alpha channels.
///
/// Surfaces store some additional properties, such as their width, height, depth, wrap mode, alpha mode, and whether they represent a normal map.
///
/// Texture data is stored non-interleaved; that is, all the red channel data is stored first, followed by the green channel data, and so on.
///
/// # Performance Notes
/// Surfaces use reference-counted pointers to image data. This means that multiple Surfaces can reference the same data. This is handled automatically by NVTT's image processing routines. For instance, after the following piece of code,
/// ```ignore
/// let s1 = Surface::image(...);
/// let mut s2 = s1.clone();
/// ```
/// surfaces `s1` and `s2` will have the same [`data()`](Surface::data) pointer. Cloning the underlying data is handled automatically: for instance, after
/// ```ignore
/// s2.to_srgb();
/// ```
/// `s2` will have a new [`data()`](Surface::data) pointer, and `s1` will be unchanged.
#[repr(transparent)]
pub struct Surface(*mut NvttSurface);

use thiserror::Error;
#[derive(Error, Clone, Copy, Debug)]
/// The error type for various [`Surface`] operations.
pub enum SurfaceError {
    #[error(
        "invalid dimenions (expected slice of length at least {expected}, found length {found})"
    )]
    InvalidDimensions { expected: u32, found: u32 },
    #[error("unknown error has occured")]
    UnknownError,
}

impl Surface {
    /// Constructs a [`Surface`] from an uncompressed byte slice of all channels. Bytes should be in native
    /// endianness.
    ///
    /// Data must be stored in `[c, z, y, x]` order. That is, channel 0's data is first, then channel 1's
    /// data, and so on. More specifically, the value of a channel `c` at texel `(x, y, z)` is at index
    /// `((c * d + z) * h + y) * w + x`. Where `w`, `h`, and `d` are input width, height, and depth
    /// of the image respectively.
    ///
    /// If the input slice is not capable of holding `w * h * d` texels in the specified format,
    /// this will return [`SurfaceError::InvalidDimensions`]. If the slice is longer than necessary,
    /// extra bytes remain unread.
    ///
    /// Formats without four R, G, B, and A channels (currently only [`InputFormat::R32f`]) will have
    /// non-existent channels filled with the color `[0, 0, 0, 0]`.
    ///
    /// # Panics
    ///
    /// Panics if any of `w`, `h`, or `d` are `0`.
    ///
    /// # Examples
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel};
    /// # use approx::assert_relative_eq;
    /// let input = InputFormat::Bgra8Ub {
    ///     data: &[0, 255, 255, 0],
    ///     unsigned_to_signed:  false,
    /// };
    /// let surface = Surface::image(input, 1, 1, 1).unwrap();
    /// assert_relative_eq!(1.0, surface.channel(Channel::R)[0]);
    /// assert_relative_eq!(1.0, surface.channel(Channel::G)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::B)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::A)[0]);
    /// ```
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel};
    /// # use approx::assert_relative_eq;
    /// let input = InputFormat::Bgra8Ub {
    ///     data: &[0, 255, 255, 0],
    ///     unsigned_to_signed:  true,
    /// };
    /// let surface = Surface::image(input, 1, 1, 1).unwrap();
    /// assert_relative_eq!(1.0, surface.channel(Channel::R)[0]);
    /// assert_relative_eq!(1.0, surface.channel(Channel::G)[0]);
    /// assert_relative_eq!(-1.0, surface.channel(Channel::B)[0]);
    /// assert_relative_eq!(-1.0, surface.channel(Channel::A)[0]);
    /// ```
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel};
    /// # use approx::assert_relative_eq;
    /// let r_bytes = 3.0_f32.to_ne_bytes();
    /// let input = InputFormat::R32f(&r_bytes);
    /// let surface = Surface::image(input, 1, 1, 1).unwrap();
    /// assert_relative_eq!(3.0, surface.channel(Channel::R)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::G)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::B)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::A)[0]);
    /// ```
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, SurfaceError};
    /// # use approx::assert_relative_eq;
    /// let input = InputFormat::Bgra8Ub {
    ///     data: &[0, 0, 0, 0],
    ///     unsigned_to_signed:  false,
    /// };
    /// let surface = Surface::image(input, 100, 100, 100);
    /// assert!(surface.is_err());
    /// ```
    pub fn image(input: InputFormat, w: u32, h: u32, d: u32) -> Result<Self, SurfaceError> {
        if !input.fit_dim(w, h, d) {
            return Err(SurfaceError::InvalidDimensions {
                expected: input.min_bytes(w, h, d),
                found: input.data().len() as u32,
            });
        }

        unsafe {
            let unsigned_to_signed = if let InputFormat::Bgra8Ub {
                unsigned_to_signed, ..
            } = input
            {
                unsigned_to_signed
            } else {
                false
            };

            let surface_ptr = nvtt_sys::nvttCreateSurface();
            if surface_ptr.is_null() {
                panic!("failed to allocate");
            }

            let ret: bool = nvtt_sys::nvttSurfaceSetImageData(
                surface_ptr,
                input.into_nvtt(),
                w as i32,
                h as i32,
                d as i32,
                input.data().as_ptr().cast(),
                unsigned_to_signed.into(),
                std::ptr::null_mut(),
            )
            .into();
            if !ret {
                nvtt_sys::nvttDestroySurface(surface_ptr);
                Err(SurfaceError::UnknownError)
            } else {
                Ok(Self(surface_ptr))
            }
        }
    }

    /// Constructs a [`Surface`] from uncompressed byte slices of all channels. Bytes should be in native endianness.
    ///
    /// Data must be stored in `[z, y, x]` order. More specifically, the value of any channel at texel `(x, y, z)`
    /// is at index `((d + z) * h + y) * w + x`. Where `w`, `h`, and `d` are input width, height, and depth of the
    /// image respectively.
    ///
    /// If any input slice is not capable of holding `w * h * d` texels in the specified format, this will return
    /// [`SurfaceError::InvalidDimensions`]. If the slice is longer than necessary, extra bytes remain unread.
    ///
    /// Formats without four R, G, B, and A channels (currently only [`SplitInputFormat::R32f`]) will have non-existent
    /// channels filled with the color `[0, 0, 0, 0]`.
    ///
    /// # Panics
    ///
    /// Panics if any of `w`, `h`, or `d` are 0.
    ///
    /// # Examples
    /// ```rust
    /// # use nvtt_rs::{Surface, SplitInputFormat, Channel};
    /// # use approx::assert_relative_eq;
    /// let input = SplitInputFormat::Bgra8Ub {
    ///     b: &[0],
    ///     g: &[255],
    ///     r: &[255],
    ///     a: &[0],
    /// };
    /// let surface = Surface::image_split(input, 1, 1, 1).unwrap();
    /// assert_relative_eq!(1.0, surface.channel(Channel::R)[0]);
    /// assert_relative_eq!(1.0, surface.channel(Channel::G)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::B)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::A)[0]);
    /// ```
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, SplitInputFormat, Channel};
    /// # use approx::assert_relative_eq;
    /// let r_bytes = 3.0_f32.to_ne_bytes();
    /// let input = SplitInputFormat::R32f(&r_bytes);
    /// let surface = Surface::image_split(input, 1, 1, 1).unwrap();
    /// assert_relative_eq!(3.0, surface.channel(Channel::R)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::G)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::B)[0]);
    /// assert_relative_eq!(0.0, surface.channel(Channel::A)[0]);
    /// ```
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, SplitInputFormat, Channel, SurfaceError};
    /// # use approx::assert_relative_eq;
    /// let input = SplitInputFormat::Bgra8Ub {
    ///     b: &[0],
    ///     g: &[0],
    ///     r: &[0],
    ///     a: &[0],
    /// };
    /// let surface = Surface::image_split(input, 100, 100, 100);
    /// assert!(surface.is_err());
    /// ```
    pub fn image_split(
        input: SplitInputFormat,
        w: u32,
        h: u32,
        d: u32,
    ) -> Result<Self, SurfaceError> {
        if !input.fit_dim(w, h, d) {
            return Err(SurfaceError::InvalidDimensions {
                expected: input.min_bytes(w, h, d),
                found: input.shortest_slice_len(),
            });
        }

        unsafe {
            let surface_ptr = nvtt_sys::nvttCreateSurface();
            if surface_ptr.is_null() {
                panic!("failed to allocate");
            }

            let (r, g, b, a) = match input {
                SplitInputFormat::Bgra8Ub { b, g, r, a } => (
                    r.as_ptr().cast(),
                    g.as_ptr().cast(),
                    b.as_ptr().cast(),
                    a.as_ptr().cast(),
                ),
                SplitInputFormat::Bgra8Sb { b, g, r, a } => (
                    r.as_ptr().cast(),
                    g.as_ptr().cast(),
                    b.as_ptr().cast(),
                    a.as_ptr().cast(),
                ),
                SplitInputFormat::Rgba32f { r, g, b, a } => (
                    r.as_ptr().cast(),
                    g.as_ptr().cast(),
                    b.as_ptr().cast(),
                    a.as_ptr().cast(),
                ),
                SplitInputFormat::Rgba16f { r, g, b, a } => (
                    r.as_ptr().cast(),
                    g.as_ptr().cast(),
                    b.as_ptr().cast(),
                    a.as_ptr().cast(),
                ),
                SplitInputFormat::R32f(r) => (
                    r.as_ptr().cast(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                ),
            };

            let ret: bool = nvtt_sys::nvttSurfaceSetImageRGBA(
                surface_ptr,
                input.into_nvtt(),
                w as i32,
                h as i32,
                d as i32,
                r,
                g,
                b,
                a,
                std::ptr::null_mut(),
            )
            .into();
            if !ret {
                nvtt_sys::nvttDestroySurface(surface_ptr);
                Err(SurfaceError::UnknownError)
            } else {
                Ok(Self(surface_ptr))
            }
        }
    }

    /// Returns the width (X size) of the surface in pixels.
    pub fn width(&self) -> u32 {
        unsafe { nvtt_sys::nvttSurfaceWidth(self.0) as u32 }
    }

    /// Returns the height (Y size) of the surface in pixels.
    pub fn height(&self) -> u32 {
        unsafe { nvtt_sys::nvttSurfaceHeight(self.0) as u32 }
    }

    /// Returns the depth (Z size) of the surface in pixels.
    pub fn depth(&self) -> u32 {
        unsafe { nvtt_sys::nvttSurfaceDepth(self.0) as u32 }
    }

    /// Returns true if data is currently held on the CPU. This is the same as checking if
    /// [`Surface::gpu_data_ptr()`] is null.
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// #[cfg(feature = "cuda")]
    /// if *CUDA_SUPPORTED {
    ///     let input_format = InputFormat::Bgra8Ub {
    ///         data: &[255, 255, 255, 255],
    ///         unsigned_to_signed: false,
    ///     };
    ///     let mut surface = Surface::image(input_format, 1, 1, 1).unwrap();
    ///     assert!(surface.on_cpu());
    ///     assert!(surface.gpu_data_ptr().is_null());
    ///
    ///     surface.to_gpu();
    ///     assert!(!surface.on_cpu());
    ///     assert!(!surface.gpu_data_ptr().is_null());
    ///
    ///     surface.to_cpu();
    ///     assert!(surface.on_cpu());
    ///     assert!(surface.gpu_data_ptr().is_null());
    /// }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    #[cfg(feature = "cuda")]
    pub fn on_cpu(&self) -> bool {
        !self.on_gpu()
    }

    /// Returns true if data is currently held to the GPU. This is the same as checking if
    /// [`Surface::gpu_data_ptr()`] is not null.
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// #[cfg(feature = "cuda")]
    /// if *CUDA_SUPPORTED {
    ///     let input_format = InputFormat::Bgra8Ub {
    ///         data: &[255, 255, 255, 255],
    ///         unsigned_to_signed: false,
    ///     };
    ///     let mut surface = Surface::image(input_format, 1, 1, 1).unwrap();
    ///     assert!(!surface.on_gpu());
    ///     assert!(surface.gpu_data_ptr().is_null());
    ///
    ///     surface.to_gpu();
    ///     assert!(surface.on_gpu());
    ///     assert!(!surface.gpu_data_ptr().is_null());
    ///
    ///     surface.to_cpu();
    ///     assert!(!surface.on_gpu());
    ///     assert!(surface.gpu_data_ptr().is_null());
    /// }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    #[cfg(feature = "cuda")]
    pub fn on_gpu(&self) -> bool {
        !self.gpu_data_ptr().is_null()
    }

    /// Copies data from CPU to GPU, enabling CUDA for all subsequent operations. Does nothing if
    /// data is already on the GPU.
    ///
    /// # Panics
    ///
    /// Panics if CUDA is not supported by this device.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// #[cfg(feature = "cuda")]
    /// if *CUDA_SUPPORTED {
    ///     let input_format = InputFormat::Bgra8Ub {
    ///         data: &[255, 255, 255, 255],
    ///         unsigned_to_signed: false,
    ///     };
    ///     let mut surface = Surface::image(input_format, 1, 1, 1).unwrap();
    ///     surface.to_gpu();
    ///     assert!(surface.on_gpu());
    /// }
    /// ```
    ///
    /// ```should_panic
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// if !*CUDA_SUPPORTED {
    ///     let input_format = InputFormat::Bgra8Ub {
    ///         data: &[255, 255, 255, 255],
    ///         unsigned_to_signed: false,
    ///     };
    ///     let mut surface = Surface::image(input_format, 1, 1, 1).unwrap();
    ///     // Panics
    ///     surface.to_gpu();
    /// }
    /// # else {
    /// #     panic!();
    /// # }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    #[cfg(feature = "cuda")]
    pub fn to_gpu(&mut self) {
        if !*CUDA_SUPPORTED {
            panic!("cuda is not supported");
        }

        if !self.on_gpu() {
            unsafe {
                nvtt_sys::nvttSurfaceToGPU(self.0, true.into(), std::ptr::null_mut());
            }
        }
    }

    /// Copies data from GPU to CPU, disabling CUDA operations. Does nothing if data is already on
    /// the CPU.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// #[cfg(feature = "cuda")]
    /// if *CUDA_SUPPORTED {
    ///     let input_format = InputFormat::Bgra8Ub {
    ///         data: &[255, 255, 255, 255],
    ///         unsigned_to_signed: false,
    ///     };
    ///     let mut surface = Surface::image(input_format, 1, 1, 1).unwrap();
    ///     surface.to_gpu();
    ///     assert!(surface.on_gpu());
    ///
    ///     surface.to_cpu();
    ///     assert!(surface.on_cpu());
    /// }
    /// ```
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// if !*CUDA_SUPPORTED {
    ///     let input_format = InputFormat::Bgra8Ub {
    ///         data: &[255, 255, 255, 255],
    ///         unsigned_to_signed: false,
    ///     };
    ///     let mut surface = Surface::image(input_format, 1, 1, 1).unwrap();
    ///     assert!(surface.on_cpu());
    ///
    ///     surface.to_cpu();
    ///     assert!(surface.on_cpu());
    /// }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    #[cfg(feature = "cuda")]
    pub fn to_cpu(&mut self) {
        if self.on_gpu() {
            unsafe {
                nvtt_sys::nvttSurfaceToCPU(self.0, std::ptr::null_mut());
            }
        }
    }

    /// Returns a CUDA pointer to image data on GPU, with the same layout as [`Surface::data()`]. If data is
    /// not currently stored on the GPU, this is a null pointer.
    ///
    /// # Safety
    ///
    /// It is undefined behaviour to derefence this pointer on the CPU, as it is not a CPU pointer.
    ///
    /// ```ignore
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// #[cfg(feature = "cuda")]
    /// if *CUDA_SUPPORTED {
    ///     let input_format = InputFormat::Bgra8Ub {
    ///         data: &[255, 255, 255, 255],
    ///         unsigned_to_signed: false,
    ///     };
    ///     let mut surface = Surface::image(input_format, 1, 1, 1).unwrap();
    ///     surface.to_gpu();
    ///
    ///     assert!(!surface.gpu_data_ptr().is_null());
    ///
    ///     // Undefined behaviour
    ///     unsafe {
    ///         let x = *surface.gpu_data_ptr();
    ///     }
    /// }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    #[cfg(feature = "cuda")]
    pub fn gpu_data_ptr(&self) -> *const f32 {
        unsafe { nvtt_sys::nvttSurfaceGPUData(self.0) }
    }

    /// Returns the value of the channel for the texel at `(x, y, z)`. This is just a wrapper
    /// around [`Surface::data()`] for easier indexing. See [`Surface::data()`] for more details.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y, z)` is out of bounds.
    pub fn texel(&self, channel: Channel, x: u32, y: u32, z: u32) -> f32 {
        if x >= self.width() || y >= self.height() || z >= self.depth() {
            panic!("texel out of bounds");
        }

        let channel = channel as i32 as u32;
        let index = ((channel * self.depth() + z) * self.height() + y) * self.width() + x;
        self.data()[index as usize]
    }

    /// Returns a mutable reference to the value of the channel for the texel at `(x, y, z)`.
    /// This is just a wrapper around [`Surface::data_mut()`] for easier indexing. See
    /// [`Surface::data_mut()`] for more details.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y, z)` is out of bounds.
    pub fn texel_mut(&mut self, channel: Channel, x: u32, y: u32, z: u32) -> &mut f32 {
        if x >= self.width() || y >= self.height() || z >= self.depth() {
            panic!("texel out of bounds");
        }

        let channel = channel as i32 as u32;
        let index = ((channel * self.depth() + z) * self.height() + y) * self.width() + x;
        &mut self.data_mut()[index as usize]
    }

    /// Returns image data as a slice in `[c, z, y, x]` order. That is, the value of a channel
    /// `c` at texel `(x, y, z)` is at index `((c * d + z) * h + y) * w + x`.
    ///
    /// # Notes
    ///
    /// If data is on the GPU, this will perform a GPU-CPU copy. CUDA will remain enabled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// # use approx::assert_relative_eq;
    /// let r_bytes = 1.0_f32.to_ne_bytes();
    /// let input = InputFormat::R32f(&r_bytes);
    /// let mut surface = Surface::image(input, 1, 1, 1).unwrap();
    ///
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if *CUDA_SUPPORTED {
    ///         surface.to_gpu();
    ///         assert!(surface.on_gpu());
    ///     } else {
    ///         assert!(surface.on_cpu());
    ///     }
    /// }
    ///
    /// // Incurs a GPU-CPU copy
    /// assert_eq!(4, surface.data().len());
    ///
    /// // Incurs another GPU-CPU copy
    /// let x = surface.data()[0];
    /// assert_relative_eq!(1.0, x);
    /// println!("{}", x);
    ///
    /// // Remains on GPU
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if *CUDA_SUPPORTED {
    ///         assert!(surface.on_gpu());
    ///     } else {
    ///         assert!(surface.on_cpu());
    ///     }
    /// }
    /// ```
    pub fn data(&self) -> &[f32] {
        unsafe {
            let len = self.width() * self.height() * self.depth() * 4;
            let ptr = nvtt_sys::nvttSurfaceData(self.0).cast_const();
            std::slice::from_raw_parts(ptr, len as usize)
        }
    }

    /// Returns image data as a mutable slice in `[c, z, y, x]` order. That is, the value of a channel
    /// `c` at texel `(x, y, z)` is at index `((c * d + z) * h + y) * w + x`.
    ///
    /// # Notes
    ///
    /// If data is on the GPU, this will perform a GPU-CPU copy. CUDA will be disabled, however it
    /// can be re-enabled with [`Surface::to_gpu()`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat};
    /// let r_bytes = 1.0_f32.to_ne_bytes();
    /// let input = InputFormat::R32f(&r_bytes);
    /// let mut surface = Surface::image(input, 1, 1, 1).unwrap();
    ///
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if *CUDA_SUPPORTED {
    ///         surface.to_gpu();
    ///         assert!(surface.on_gpu());
    ///     } else {
    ///         assert!(surface.on_cpu());
    ///     }
    /// }
    ///
    /// // Moves to CPU
    /// surface.data_mut()[0] = 0.0;
    ///
    /// #[cfg(feature = "cuda")]
    /// assert!(surface.on_cpu());
    /// ```
    pub fn data_mut(&mut self) -> &mut [f32] {
        cfg_if::cfg_if! {
            if #[cfg(feature = "cuda")] {
                self.to_cpu();
            }
        }

        unsafe {
            let len = self.width() * self.height() * self.depth() * 4;
            let ptr = nvtt_sys::nvttSurfaceData(self.0);
            std::slice::from_raw_parts_mut(ptr, len as usize)
        }
    }

    /// Returns a slice of image data for `channel` in `[z, y, x]` order. That is, the value of
    /// texel `(x, y, z)` is at index `((d + z) * h + y) * w + x`.
    ///
    /// # Notes
    ///
    /// If data is on the GPU, this will perform a GPU-CPU copy. CUDA will remain enabled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat, Channel};
    /// # use approx::assert_relative_eq;
    /// let r_bytes = 1.0_f32.to_ne_bytes();
    /// let input = InputFormat::R32f(&r_bytes);
    /// let mut surface = Surface::image(input, 1, 1, 1).unwrap();
    ///
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if *CUDA_SUPPORTED {
    ///         surface.to_gpu();
    ///         assert!(surface.on_gpu());
    ///     } else {
    ///         assert!(surface.on_cpu());
    ///     }
    /// }
    ///
    /// // Each call incurs a GPU-CPU copy
    /// assert_eq!(1, surface.channel(Channel::R).len());
    /// assert_eq!(1, surface.channel(Channel::G).len());
    /// assert_eq!(1, surface.channel(Channel::B).len());
    /// assert_eq!(1, surface.channel(Channel::A).len());
    ///
    /// // Incurs another GPU-CPU copy
    /// let x = surface.channel(Channel::R)[0];
    /// println!("{}", x);
    /// assert_relative_eq!(1.0, x);
    ///
    /// // Remains on GPU
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if *CUDA_SUPPORTED {
    ///         surface.to_gpu();
    ///         assert!(surface.on_gpu());
    ///     } else {
    ///         assert!(surface.on_cpu());
    ///     }
    /// }
    /// ```
    pub fn channel(&self, channel: Channel) -> &[f32] {
        let channel = channel as i32;

        unsafe {
            let len = self.width() * self.height() * self.depth();
            let ptr = nvtt_sys::nvttSurfaceChannel(self.0, channel).cast_const();
            std::slice::from_raw_parts(ptr, len as usize)
        }
    }

    /// Returns a mutable slice of image data for `channel` in `[z, y, x]` order. That is, the value of
    /// texel `(x, y, z)` is at index `((d + z) * h + y) * w + x`.
    ///
    /// # Notes
    ///
    /// If data is on the GPU, this will perform a GPU-CPU copy. CUDA will be disabled, however it
    /// can be re-enabled with [`Surface::to_gpu()`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{CUDA_SUPPORTED, Surface, InputFormat, Channel};
    /// let r_bytes = 1.0_f32.to_ne_bytes();
    /// let input = InputFormat::R32f(&r_bytes);
    /// let mut surface = Surface::image(input, 1, 1, 1).unwrap();
    ///
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if *CUDA_SUPPORTED {
    ///         surface.to_gpu();
    ///         assert!(surface.on_gpu());
    ///     } else {
    ///         assert!(surface.on_cpu());
    ///     }
    /// }
    ///
    /// // Moves to CPU
    /// surface.channel_mut(Channel::R)[0] = 0.0;
    ///
    /// #[cfg(feature = "cuda")]
    /// assert!(surface.on_cpu());
    /// ```
    pub fn channel_mut(&mut self, channel: Channel) -> &mut [f32] {
        cfg_if::cfg_if! {
            if #[cfg(feature = "cuda")] {
                self.to_cpu();
            }
        }

        let channel = channel as i32;

        unsafe {
            let len = self.width() * self.height() * self.depth();
            let ptr = nvtt_sys::nvttSurfaceChannel(self.0, channel);
            std::slice::from_raw_parts_mut(ptr, len as usize)
        }
    }

    /// Returns [`TextureType::D2`] if `depth == 1`, and [`TextureType::D3`] otherwise.
    pub fn tex_type(&self) -> TextureType {
        unsafe { nvtt_sys::nvttSurfaceType(self.0).into() }
    }

    /// Returns the [`WrapMode`] of this surface. Note that [`WrapMode`] can affect operations such as
    ///  [`Surface::build_next_mipmap()`] or [`Surface::convolve()`].
    pub fn wrap_mode(&self) -> WrapMode {
        unsafe { nvtt_sys::nvttSurfaceWrapMode(self.0).into() }
    }

    /// Returns the [`AlphaMode`] of this surface. This is for output headers, e.g. for usage in [`Context::output_header`].
    /// It does not affect calculations.
    pub fn alpha_mode(&self) -> AlphaMode {
        unsafe { nvtt_sys::nvttSurfaceAlphaMode(self.0).into() }
    }

    /// Returns the true if this surface represents a normal map. This is for output headers,
    /// e.g. for usage in [`Context::output_header`]. It does not affect calculations.
    pub fn is_normal_map(&self) -> bool {
        unsafe { nvtt_sys::nvttSurfaceIsNormalMap(self.0).into() }
    }

    ///  Set the [`WrapMode`] of this surface. Note that [`WrapMode`] can affect operations such as
    ///  [`Surface::build_next_mipmap()`] or [`Surface::convolve()`].
    pub fn set_wrap_mode(&mut self, wrap_mode: WrapMode) {
        unsafe {
            nvtt_sys::nvttSetSurfaceWrapMode(self.0, wrap_mode.into());
        }
    }

    /// Set the [`AlphaMode`] of this surface. This is for output headers, e.g. for usage in [`Context::output_header`].
    /// It does not affect calculations.
    pub fn set_alpha_mode(&mut self, alpha_mode: AlphaMode) {
        unsafe {
            nvtt_sys::nvttSetSurfaceAlphaMode(self.0, alpha_mode.into());
        }
    }

    /// Set whether this surface represents a normal map. This is for output headers,
    /// e.g. for usage in [`Context::output_header`]. It does not affect calculations.
    pub fn set_normal_map(&mut self, is_normal_map: bool) {
        unsafe {
            nvtt_sys::nvttSetSurfaceNormalMap(self.0, is_normal_map.into());
        }
    }

    /// Flip along the X axis.
    pub fn flip_x(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceFlipX(self.0, std::ptr::null_mut());
        }
    }

    /// Flip along the Y axis.
    pub fn flip_y(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceFlipY(self.0, std::ptr::null_mut());
        }
    }

    /// Flip along the Z axis.
    pub fn flip_z(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceFlipZ(self.0, std::ptr::null_mut());
        }
    }

    /// Copy `src_channel` from `other` to `dst_channel` of this surface. If channels are
    /// not the same length, this returns [`SurfaceError::InvalidDimensions`]
    pub fn copy_channel(
        &mut self,
        other: &Self,
        src_channel: Channel,
        dst_channel: Channel,
    ) -> Result<(), SurfaceError> {
        unsafe {
            if nvtt_sys::nvttSurfaceCopyChannel(
                self.0,
                other.0,
                src_channel as i32,
                dst_channel as i32,
                std::ptr::null_mut(),
            )
            .into()
            {
                Ok(())
            } else {
                Err(SurfaceError::InvalidDimensions {
                    expected: self.channel(dst_channel).len() as u32,
                    found: other.channel(src_channel).len() as u32,
                })
            }
        }
    }

    /// Add `src_channel` from `other` multiplied by `scale` to `dst_channel` of this surface.
    /// If channels are not the same length, this returns [`SurfaceError::InvalidDimensions`]
    pub fn add_channel(
        &mut self,
        other: &Self,
        src_channel: Channel,
        dst_channel: Channel,
        scale: f32,
    ) -> Result<(), SurfaceError> {
        unsafe {
            if nvtt_sys::nvttSurfaceAddChannel(
                self.0,
                other.0,
                src_channel as i32,
                dst_channel as i32,
                scale,
                std::ptr::null_mut(),
            )
            .into()
            {
                Ok(())
            } else {
                Err(SurfaceError::InvalidDimensions {
                    expected: self.channel(dst_channel).len() as u32,
                    found: other.channel(src_channel).len() as u32,
                })
            }
        }
    }

    /// Returns the [`BoundingBox`] of this surface. `min_x/min_y/min_z` are all `0`, while
    /// `max_x`, `min_y`, and `min_z` are [`Surface::width()`], [`Surface::height()`], and
    /// [`Surface::depth()`] respectively.
    pub fn bounds(&self) -> BoundingBox {
        BoundingBox {
            min_x: 0,
            max_x: self.width(),
            min_y: 0,
            max_y: self.height(),
            min_z: 0,
            max_z: self.depth(),
        }
    }

    /// Copies the bounding box `src` of `other` to the bounding box `dst` of this surface.
    ///
    /// # Panics
    ///
    /// Panics if `src` and `dst` are not the same shape, or one of the bounding boxes is out of bounds.
    pub fn copy(&mut self, other: &Self, src: BoundingBox, dst: BoundingBox) {
        if !other.bounds().contains(dst) || !self.bounds().contains(dst) || !src.same_shape(dst) {
            panic!("invalid bounding boxes supplied");
        } else {
            let xsrc = src.min_x as i32;
            let ysrc = src.min_y as i32;
            let zsrc = src.min_z as i32;

            let xsize = (dst.max_x - dst.min_x) as i32;
            let ysize = (dst.max_y - dst.min_y) as i32;
            let zsize = (dst.max_z - dst.min_z) as i32;

            let xdst = dst.min_x as i32;
            let ydst = dst.min_y as i32;
            let zdst = dst.min_z as i32;

            unsafe {
                nvtt_sys::nvttSurfaceCopy(
                    self.0,
                    other.0,
                    xsrc,
                    ysrc,
                    zsrc,
                    xsize,
                    ysize,
                    zsize,
                    xdst,
                    ydst,
                    zdst,
                    std::ptr::null_mut(),
                );
            }
        }
    }

    /// Creates a sub image from the subset `bounds` of this surface.
    ///
    /// # Panics
    ///
    /// Panics if this surface does not contain `bounds`.
    pub fn create_sub_image(&self, bounds: BoundingBox) -> Self {
        if !self.bounds().contains(bounds) {
            panic!("invalid bounds supplied");
        } else {
            let x0 = bounds.min_x as i32;
            let x1 = bounds.max_x as i32;
            let y0 = bounds.min_y as i32;
            let y1 = bounds.max_y as i32;
            let z0 = bounds.min_z as i32;
            let z1 = bounds.max_z as i32;

            unsafe {
                let ptr = nvtt_sys::nvttSurfaceCreateSubImage(
                    self.0,
                    x0,
                    x1,
                    y0,
                    y1,
                    z0,
                    z1,
                    std::ptr::null_mut(),
                );

                if ptr.is_null() {
                    panic!("failed to allocate");
                } else {
                    Self(ptr)
                }
            }
        }
    }

    /// Returns the number of mipmaps in a full mipmap chain. Each mip is half the size of the previous,
    /// rounding down, until and including a `1x1` mip.
    ///
    /// For instance, a 8x5 surface has mipmaps of size 8x5 (mip 0), 4x2 (mip 1), 2x1 (mip 2), and 1x1 (mip 3),
    /// so [`Surface::count_mipmaps()`] returns 4.
    /// A 7x3 surface has mipmaps of size 7x3, 3x1, and 1x1, so [`Surface::count_mipmaps()`] returns 3.
    ///
    /// Same as [`Surface::count_mipmaps_until()`] with `min_size == 1`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel};
    /// let bytes = [0u8; (8 * 5 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    ///
    /// let surface = Surface::image(input, 8, 5, 1).unwrap();
    /// assert_eq!(4, surface.count_mipmaps())
    /// ```
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel};
    /// let bytes = [0u8; (7 * 3 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    ///
    /// let surface = Surface::image(input, 7, 3, 1).unwrap();
    /// assert_eq!(3, surface.count_mipmaps())
    /// ```
    pub fn count_mipmaps(&self) -> u32 {
        unsafe { nvtt_sys::nvttSurfaceCountMipmaps(self.0, 1) as u32 }
    }

    /// Returns the number of mipmaps in a mipmap chain, stopping when
    /// [`Surface::can_make_next_mipmap()`] returns false.  
    ///
    /// That is, it stops when a `1x1x1` mip is reached if `min_size == 1`
    /// (in which case it is the same as [`Surface::count_mipmaps()`]),
    /// or stops when the width and height are less than `min_size` and the depth is `1`.
    pub fn count_mipmaps_until(&self, min_size: u32) -> u32 {
        unsafe { nvtt_sys::nvttSurfaceCountMipmaps(self.0, min_size as i32) as u32 }
    }

    /// Returns whether the surface would have a next mip in a mip chain with minimum size `min_size`.
    ///
    /// That is, it returns false if this surface has size `1x1x1`, or if the width and height are less
    /// than `min_size` and the depth is 1.
    pub fn can_make_next_mipmap(&self, min_size: u32) -> bool {
        unsafe { nvtt_sys::nvttSurfaceCanMakeNextMipmap(self.0, min_size as i32).into() }
    }

    /// Resizes this surface to create the next mip in a mipmap chain.
    ///
    /// Returns false iff the next mip would have been smaller than `min_size`
    /// (signaling the end of the mipmap chain).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// let bytes = [0u8; (32 * 32 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    /// let mut surface = Surface::image(input, 32, 32, 1).unwrap();
    ///
    /// let mipmap_count = surface.count_mipmaps();
    /// let mut mipmaps = vec![surface.clone()];
    /// while surface.build_next_mipmap(Filter::mipmap_box(), 1) {
    ///     mipmaps.push(surface.clone());
    /// }
    ///
    /// assert_eq!(mipmap_count, mipmaps.len() as u32);
    ///
    /// assert_eq!(mipmaps[0].width(), mipmaps[0].height());
    /// assert_eq!(mipmaps[1].width(), mipmaps[1].height());
    /// assert_eq!(mipmaps[2].width(), mipmaps[2].height());
    /// assert_eq!(mipmaps[3].width(), mipmaps[3].height());
    /// assert_eq!(mipmaps[4].width(), mipmaps[4].height());
    /// assert_eq!(mipmaps[5].width(), mipmaps[5].height());
    ///
    /// assert_eq!(32, mipmaps[0].width());
    /// assert_eq!(16, mipmaps[1].width());
    /// assert_eq!(8, mipmaps[2].width());
    /// assert_eq!(4, mipmaps[3].width());
    /// assert_eq!(2, mipmaps[4].width());
    /// assert_eq!(1, mipmaps[5].width());
    /// ```
    pub fn build_next_mipmap(&mut self, filter: Filter<Mipmap>, min_size: u32) -> bool {
        let filter_width = filter.width;
        let params = filter.params();
        let params_ptr = filter.params_ptr(&params);

        unsafe {
            nvtt_sys::nvttSurfaceBuildNextMipmap(
                self.0,
                filter.algorithm.into(),
                filter_width,
                params_ptr,
                min_size as i32,
                std::ptr::null_mut(),
            )
            .into()
        }
    }

    /// Replaces this surface with a surface the size of the next mip in a mip chain
    /// (half the width and height), but with each channel cleared to a constant value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let bytes = [0u8; (32 * 32 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    /// let mut surface = Surface::image(input, 32, 32, 1).unwrap();
    ///
    /// let rgba = [0.25, 0.5, 0.75, 1.0];
    /// surface.build_next_mipmap_color(rgba);
    ///
    /// assert_eq!(16, surface.width());
    /// assert_eq!(16, surface.height());
    ///
    /// for x in 0..16 {
    ///     for y in 0..16 {
    ///         assert_relative_eq!(0.25, surface.texel(Channel::R, x, y, 0));
    ///         assert_relative_eq!(0.5,  surface.texel(Channel::G, x, y, 0));
    ///         assert_relative_eq!(0.75, surface.texel(Channel::B, x, y, 0));
    ///         assert_relative_eq!(1.0,  surface.texel(Channel::A, x, y, 0));
    ///     }
    /// }
    /// ```
    pub fn build_next_mipmap_color(&mut self, rgba: [f32; 4]) -> bool {
        unsafe {
            nvtt_sys::nvttSurfaceBuildNextMipmapSolidColor(
                self.0,
                rgba.as_ptr(),
                std::ptr::null_mut(),
            )
            .into()
        }
    }

    /// Sets all texels in the surface to a solid color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let bytes = [0u8; (32 * 32 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    /// let mut surface = Surface::image(input, 32, 32, 1).unwrap();
    ///
    /// let rgba = [0.0, 0.33, 0.66, 1.0];
    /// surface.fill(rgba);
    ///
    /// assert_eq!(32, surface.width());
    /// assert_eq!(32, surface.height());
    ///
    /// for x in 0..32 {
    ///     for y in 0..32 {
    ///         assert_relative_eq!(0.0,  surface.texel(Channel::R, x, y, 0));
    ///         assert_relative_eq!(0.33, surface.texel(Channel::G, x, y, 0));
    ///         assert_relative_eq!(0.66, surface.texel(Channel::B, x, y, 0));
    ///         assert_relative_eq!(1.0,  surface.texel(Channel::A, x, y, 0));
    ///     }
    /// }
    /// ```
    pub fn fill(&mut self, rgba: [f32; 4]) {
        let r = rgba[0];
        let g = rgba[1];
        let b = rgba[2];
        let a = rgba[3];

        unsafe { nvtt_sys::nvttSurfaceFill(self.0, r, g, b, a, std::ptr::null_mut()) }
    }

    /// Sets all texels on the border of the surface to a solid color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let bytes = [0u8; (4 * 4 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    /// let mut surface = Surface::image(input, 4, 4, 1).unwrap();
    ///
    /// let rgba = [0.25, 0.5, 0.75, 1.0];
    /// surface.set_border(rgba);
    ///
    /// for x in 0..4 {
    ///     for y in 0..4 {
    ///         // On the border
    ///         if x == 0 || y == 0 || x == 3 || y == 3 {
    ///             assert_relative_eq!(0.25, surface.texel(Channel::R, x, y, 0));
    ///             assert_relative_eq!(0.5,  surface.texel(Channel::G, x, y, 0));
    ///             assert_relative_eq!(0.75, surface.texel(Channel::B, x, y, 0));
    ///             assert_relative_eq!(1.0,  surface.texel(Channel::A, x, y, 0));
    ///         }
    ///
    ///         // Not on the border
    ///         else {
    ///             assert_relative_eq!(f32::from_bits(0), surface.texel(Channel::R, x, y, 0));
    ///             assert_relative_eq!(f32::from_bits(0), surface.texel(Channel::G, x, y, 0));
    ///             assert_relative_eq!(f32::from_bits(0), surface.texel(Channel::B, x, y, 0));
    ///             assert_relative_eq!(f32::from_bits(0), surface.texel(Channel::A, x, y, 0));
    ///         }
    ///     }
    /// }
    /// ```
    pub fn set_border(&mut self, rgba: [f32; 4]) {
        let r = rgba[0];
        let g = rgba[1];
        let b = rgba[2];
        let a = rgba[3];

        unsafe { nvtt_sys::nvttSurfaceSetBorder(self.0, r, g, b, a, std::ptr::null_mut()) }
    }

    /// Draws borders of a given color around each w x h tile contained within the surface,
    /// starting from the (0, 0) corner. In case the surface size is not divisible by the
    /// tile size, borders are not drawn for tiles crossing the surface boundary
    ///
    /// # Panics
    ///
    /// Panics if `w` or `h` are `0`.
    pub fn set_atlas_border(&mut self, w: u32, h: u32, rgba: [f32; 4]) {
        if w == 0 || h == 0 {
            panic!("invalid atlas dimensions");
        }

        let r = rgba[0];
        let g = rgba[1];
        let b = rgba[2];
        let a = rgba[3];

        unsafe {
            nvtt_sys::nvttSurfaceSetAtlasBorder(
                self.0,
                w as i32,
                h as i32,
                r,
                g,
                b,
                a,
                std::ptr::null_mut(),
            );
        }
    }

    /// Resizes this surface to have size w x h x d using a given filter.
    ///
    /// # Panics
    ///
    /// Panics if `w`, `h`, or `d` are 0
    pub fn resize_filtered(&mut self, w: u32, h: u32, d: u32, filter: Filter<Resize>) {
        if w == 0 || h == 0 || d == 0 {
            panic!("invalid resize dimensions");
        }

        let filter_width = filter.width;
        let params = filter.params();
        let params_ptr = filter.params_ptr(&params);

        unsafe {
            nvtt_sys::nvttSurfaceResize(
                self.0,
                w as i32,
                h as i32,
                d as i32,
                filter.algorithm.into(),
                filter_width,
                params_ptr,
                std::ptr::null_mut(),
            )
        }
    }

    /// Resizes this surface so that its largest side has length `max_extent`, subject to a rounding mode and a filter.
    ///
    /// # Panics
    ///
    /// Panics if `max_extent` is `0`.
    pub fn resize_rounded(&mut self, max_extent: u32, mode: RoundMode, filter: Filter<Resize>) {
        if max_extent == 0 {
            panic!("invalid max extent");
        }

        let filter_width = filter.width;
        let params = filter.params();
        let params_ptr = filter.params_ptr(&params);

        unsafe {
            nvtt_sys::nvttSurfaceResizeMaxParams(
                self.0,
                max_extent as i32,
                mode.into(),
                filter.algorithm.into(),
                filter_width,
                params_ptr,
                std::ptr::null_mut(),
            )
        }
    }

    /// Resizes this surface so that its largest side has length `max_extent` and the result is square or cubical.
    /// Uses a rounding mode and a filter.
    ///
    /// # Panics
    ///
    /// Panics if `max_extent` is `0`.
    pub fn resize_make_square(&mut self, max_extent: u32, mode: RoundMode, filter: Filter<Resize>) {
        if max_extent == 0 {
            panic!("invalid max extent");
        }

        unsafe {
            nvtt_sys::nvttSurfaceResizeMakeSquare(
                self.0,
                max_extent as i32,
                mode.into(),
                filter.algorithm.into(),
                std::ptr::null_mut(),
            );
        }
    }

    /// Crops or expands this surface from the `(0, 0, 0)` corner, with any new values cleared to 0.
    ///
    /// # Panics
    ///
    /// Panics if any of `w`, `h`, or `d` are `0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let bytes = [0u8; (4 * 4 * std::mem::size_of::<f32>())];
    /// let input = InputFormat::R32f(&bytes);
    /// let mut surface = Surface::image(input, 4, 4, 1).unwrap();
    ///
    /// let rgba = [1.0, 1.0, 1.0, 1.0];
    /// surface.fill(rgba);
    /// surface.canvas_resize(8, 8, 1);
    ///
    /// for x in 0..8 {
    ///     for y in 0..8 {
    ///         // Old region
    ///         if x < 4 && y < 4 {
    ///             assert_relative_eq!(1.0, surface.texel(Channel::R, x, y, 0));
    ///             assert_relative_eq!(1.0, surface.texel(Channel::G, x, y, 0));
    ///             assert_relative_eq!(1.0, surface.texel(Channel::B, x, y, 0));
    ///             assert_relative_eq!(1.0, surface.texel(Channel::A, x, y, 0));
    ///         }
    ///
    ///         // New region
    ///         else {
    ///             assert_relative_eq!(0.0, surface.texel(Channel::R, x, y, 0));
    ///             assert_relative_eq!(0.0, surface.texel(Channel::G, x, y, 0));
    ///             assert_relative_eq!(0.0, surface.texel(Channel::B, x, y, 0));
    ///             assert_relative_eq!(0.0, surface.texel(Channel::A, x, y, 0));
    ///         }
    ///     }
    /// }
    /// ```
    pub fn canvas_resize(&mut self, w: u32, h: u32, d: u32) {
        if w == 0 || h == 0 || d == 0 {
            panic!("invalid canvas dimensions");
        }

        unsafe {
            nvtt_sys::nvttSurfaceCanvasSize(
                self.0,
                w as i32,
                h as i32,
                d as i32,
                std::ptr::null_mut(),
            );
        }
    }

    /// Converts to premultiplied alpha, replacing `(r, g, b, a)` with `(ar, ag, ab, a)`.
    ///
    /// # Examples
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let input = InputFormat::Bgra8Ub {
    ///     data: &[255, 255, 255, 0],
    ///     unsigned_to_signed: true,
    /// };
    /// let mut surface = Surface::image(input, 1, 1, 1).unwrap();
    /// assert_relative_eq!(1.0,  surface.texel(Channel::R, 0, 0, 0));
    /// assert_relative_eq!(1.0,  surface.texel(Channel::G, 0, 0, 0));
    /// assert_relative_eq!(1.0,  surface.texel(Channel::B, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::A, 0, 0, 0));
    ///
    /// surface.premultiply_alpha();
    /// assert_relative_eq!(-1.0, surface.texel(Channel::R, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::G, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::B, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::A, 0, 0, 0));
    /// ```
    pub fn premultiply_alpha(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfacePremultiplyAlpha(self.0, std::ptr::null_mut());
        }
    }

    /// Converts from premultiplied to unpremultiplied alpha, with special handling around zero alpha values.
    ///
    /// When `abs(a) >= epsilon`, the result is the same as dividing the RGB channels by the alpha channel.
    /// Otherwise, this function divides the RGB channels by `epsilon * sign(a)`, since the result of
    /// unpremultiplying a fully transparent color is undefined.
    ///
    /// # Optional Parameters
    /// - `epsilon`: Defaults to `1e-12`
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is `0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let input = InputFormat::Bgra8Ub {
    ///     data: &[0, 0, 0, 0],
    ///     unsigned_to_signed: true,
    /// };
    /// let mut surface = Surface::image(input, 1, 1, 1).unwrap();
    /// assert_relative_eq!(-1.0, surface.texel(Channel::R, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::G, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::B, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::A, 0, 0, 0));
    ///
    /// surface.demultiply_alpha(None);
    /// assert_relative_eq!(1.0,  surface.texel(Channel::R, 0, 0, 0));
    /// assert_relative_eq!(1.0,  surface.texel(Channel::G, 0, 0, 0));
    /// assert_relative_eq!(1.0,  surface.texel(Channel::B, 0, 0, 0));
    /// assert_relative_eq!(-1.0, surface.texel(Channel::A, 0, 0, 0));
    /// ```
    pub fn demultiply_alpha(&mut self, epsilon: Option<f32>) {
        let epsilon = epsilon.unwrap_or(1e-12);
        if epsilon == 0.0 {
            panic!("epsilon must be nonzero");
        }

        unsafe {
            nvtt_sys::nvttSurfaceDemultiplyAlpha(self.0, epsilon, std::ptr::null_mut());
        }
    }

    /// Sets channels to the result of converting to grayscale, with customizable channel weights.
    ///
    /// For instance, this can be used to give green a higher weight than red or blue when computing luminance.
    /// This function will normalize the different scales so they sum to 1, so e.g. (2, 4, 1, 0) are valid
    /// scales. The greyscale value is then computed using
    ///
    /// `grey = r * rgba_scale[0] + g * rgba_scale[1] + b * rgba_scale[2] + a * rgba_scale[3]`
    ///
    /// and then all channels (including alpha) are set to grey.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use nvtt_rs::{Surface, InputFormat, Channel, Filter};
    /// # use approx::assert_relative_eq;
    /// let input = InputFormat::Bgra8Ub {
    ///     data: &[53, 234, 26, 158],
    ///     unsigned_to_signed: false,
    /// };
    /// let mut surface = Surface::image(input, 1, 1, 1).unwrap();
    /// surface.to_grey_scale([2.0, 4.0, 1.0, 0.0]);
    ///
    /// let r = surface.texel(Channel::R, 0, 0, 0);
    /// let g = surface.texel(Channel::G, 0, 0, 0);
    /// let b = surface.texel(Channel::B, 0, 0, 0);
    /// let a = surface.texel(Channel::A, 0, 0, 0);
    ///
    /// assert_relative_eq!(r, g);
    /// assert_relative_eq!(g, b);
    /// assert_relative_eq!(b, a);
    /// ```
    pub fn to_grey_scale(&mut self, rgba_scale: [f32; 4]) {
        let r_scale = rgba_scale[0];
        let g_scale = rgba_scale[1];
        let b_scale = rgba_scale[2];
        let a_scale = rgba_scale[3];

        unsafe {
            nvtt_sys::nvttSurfaceToGreyScale(
                self.0,
                r_scale,
                g_scale,
                b_scale,
                a_scale,
                std::ptr::null_mut(),
            )
        }
    }

    /// Raises RGB channels to the power `gamma`. `gamma=2.2` approximates sRGB-to-linear conversion.
    pub fn from_gamma(&mut self, gamma: f32) {
        unsafe { nvtt_sys::nvttSurfaceToLinear(self.0, gamma, std::ptr::null_mut()) }
    }

    /// Raises the given channel to the power `gamma`.
    pub fn channel_from_gamma(&mut self, channel: Channel, gamma: f32) {
        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceToLinearChannel(self.0, channel, gamma, std::ptr::null_mut())
        }
    }

    /// Raises RGB channels to the power `1/gamma`. `gamma=2.2` approximates sRGB-to-linear conversion.
    pub fn to_gamma(&mut self, gamma: f32) {
        unsafe { nvtt_sys::nvttSurfaceToGamma(self.0, gamma, std::ptr::null_mut()) }
    }

    /// Raises the given channel to the power `1/gamma`.
    pub fn channel_to_gamma(&mut self, channel: Channel, gamma: f32) {
        let channel = channel as i32;

        unsafe { nvtt_sys::nvttSurfaceToGammaChannel(self.0, channel, gamma, std::ptr::null_mut()) }
    }

    /// Applies the linear-to-sRGB transfer function to RGB channels.
    ///
    /// This transfer function replaces each value x with
    ///
    /// ```text
    /// if x is NaN or x <= 0.0f, 0.0f
    /// if x <= 0.0031308f, 12.92f * x
    /// if x <= 1.0f, powf(x, 0.41666f) * 1.055f - 0.055f
    /// otherwise, 1.0f
    /// ```
    pub fn to_srgb(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceToSrgb(self.0, std::ptr::null_mut());
        }
    }

    /// Applies the sRGB-to-linear transfer function to RGB channels.
    ///
    /// This transfer function replaces each value x with
    ///
    /// ```text
    /// if x < 0.0f, 0.0f
    /// if x < 0.04045f, x / 12.92f
    /// if x < 1.0f, powf((x + 0.055f)/1.055f, 2.4f)
    /// otherwise, 1.0f
    /// ```
    pub fn from_srgb(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceToLinearFromSrgb(self.0, std::ptr::null_mut());
        }
    }

    /// Converts RGB channels from linear to a piecewise linear sRGB approximation.
    ///
    /// This transfer function replaces each value x with
    ///
    /// ```text
    /// if x < 0, 0.0f
    /// if x < 1/16, 4.0f * x
    /// if x < 1/8, 0.25f + 2.0f * (x - 0.0625f)
    /// if x < 1/2, 0.375f + (x - 0.125f)
    /// if x < 1, 0.75f + 0.5f * (x - 0.5f)
    /// otherwise, 1.0f
    /// ```
    pub fn to_xenon_srgb(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceToXenonSrgb(self.0, std::ptr::null_mut());
        }
    }

    /// Produces an LDR Red, Green, Blue, Magnitude encoding of the HDR RGB channels. See
    /// [`Surface::from_rgbm()`] for the storage method. This uses an iterative compression
    /// approach to reduce the error with regard to decoding.
    ///
    /// # Optional Parameters
    /// - `range`: Defaults to `1.0`
    /// - `threshold`: Defaults to `0.25`
    pub fn to_rgbm(&mut self, range: Option<f32>, threshold: Option<f32>) {
        let range = range.unwrap_or(1.0);
        let threshold = threshold.unwrap_or(0.25);

        unsafe {
            nvtt_sys::nvttSurfaceToRGBM(self.0, range, threshold, std::ptr::null_mut());
        }
    }

    /// Produces HDR `(r, g, b, 1)` values from an LDR `(red, green, blue, magnitude)` storage method.
    ///
    /// HDR values are reconstructed as follows: First, the magnitude `M` is reconstructed from the
    /// alpha channel using `M = a * (range - threshold) + threshold`. Then the red, green, and blue
    /// channels are multiplied by `M`.
    ///
    /// # Optional Parameters
    /// - `range`: Defaults to `1.0`
    /// - `threshold`: Defaults to `0.25`
    pub fn from_rgbm(&mut self, range: Option<f32>, threshold: Option<f32>) {
        let range = range.unwrap_or(1.0);
        let threshold = threshold.unwrap_or(0.25);

        unsafe {
            nvtt_sys::nvttSurfaceFromRGBM(self.0, range, threshold, std::ptr::null_mut());
        }
    }

    /// Applies an HDR-to-LDR tone mapper.
    pub fn tonemap(&mut self, tm: ToneMapper) {
        // Unused
        let parameters = std::ptr::null_mut();

        unsafe {
            nvtt_sys::nvttSurfaceToneMap(self.0, tm.into(), parameters, std::ptr::null_mut());
        }
    }

    /// Produces a shared-exponent Red, Green, Blue, Exponent encoding of the HDR RGB channels,
    /// such as `R9G9B9E5`.
    ///
    /// See [`Surface::from_rgbe()`] for the storage method. This uses an iterative compression approach to
    /// reduce the error with regard to decoding.
    ///
    /// # Safety
    ///
    /// The Nvidia SDK does not declare what a safe value for `mantissa_bits`/`exponent_bits` may be.
    /// If they are too large or small, it will cause undefined behaviour. Assuming the operation has not changed
    /// much since the original [open sourced implementation](https://github.com/castano/nvidia-texture-tools/blob/master/src/nvtt/Surface.cpp),
    /// then `mantissa_bits`/`exponent_bits` cannot be larger than 32, as they will both be casted to a `c_int`
    /// and then perform the bitshifts `1 << mantissa_bits` and `1 << exponent_bits`. No current version lower/upper bounds are formally stated.
    ///
    /// Only the values `mantissa_bits = 9` and `exponent_bits = 5` are mentioned, however a far
    /// broader range of values is likely safe.
    pub unsafe fn to_rgbe(&mut self, mantissa_bits: u32, exponent_bits: u32) {
        unsafe {
            nvtt_sys::nvttSurfaceToRGBE(
                self.0,
                mantissa_bits as i32,
                exponent_bits as i32,
                std::ptr::null_mut(),
            );
        }
    }

    /// Produces HDR `(r, g, b, 1)` values from an LDR `(red, green, blue, exponent)` storage method.
    ///
    /// HDR values are reconstructed as follows: R, G, B, and E are first converted from `UNORM` floats
    /// to integers by multiplying RGB by `(1 << mantissaBits) - 1` and E by `(1 << exponentBits) - 1`.
    /// E stores a scaling factor as a power of 2, which is reconstructed using
    /// `scale = 2^(E - ((1 << (exponentBits - 1)) - 1) - mantissaBits)`.
    /// R, G, and B are then multiplied by `scale`.
    ///
    /// # Safety
    ///
    /// The Nvidia SDK does not declare what a safe value for `mantissa_bits`/`exponent_bits` may be.
    /// If they are too large or small, it will cause undefined behaviour. Assuming the operation has not changed
    /// much since the original [open sourced implementation](https://github.com/castano/nvidia-texture-tools/blob/master/src/nvtt/Surface.cpp),
    /// then `mantissa_bits`/`exponent_bits` cannot be larger than 32, as they will both be casted to a `c_int`
    /// and then perform the bitshifts `1 << mantissa_bits` and `1 << exponent_bits`. No current version lower/upper bounds are formally stated.
    ///
    /// Only the values `mantissa_bits = 9` and `exponent_bits = 5` are mentioned, however a far
    /// broader range of values is likely safe.
    pub unsafe fn from_rgbe(&mut self, mantissa_bits: u32, exponent_bits: u32) {
        unsafe {
            nvtt_sys::nvttSurfaceFromRGBE(
                self.0,
                mantissa_bits as i32,
                exponent_bits as i32,
                std::ptr::null_mut(),
            );
        }
    }

    /// Converts from `(r, g, b, -)` colors to `(Co, Cg, 1, Y)` colors.
    ///
    /// This is useful for formats that use chroma subsampling.
    ///
    /// Y is in the range `[0, 1]`, while Co and Cg are in the range `[-1, 1]`.
    ///
    /// The RGB-to-YCoCg formula used is
    ///
    /// ```text
    /// Y  = (2g + r + b)/4
    /// Co = r - b
    /// Cg = (2g - r - b)/2
    /// ```
    pub fn to_ycocg(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceToYCoCg(self.0, std::ptr::null_mut());
        }
    }

    /// Stores per-block YCoCg scaling information for potentially better 4-channel compression of YCoCg data.
    ///
    /// For each 4x4 block, this computes the maximum absolute Co and Cg values, stores the result in the blue
    /// channel, and multiplies the Co and Cg channels (0 and 1) by its reciprocal. The original Co and Cg values
    /// can then be reconstructed by multiplying by the blue channel.
    ///
    /// The scaling information is quantized to the given number of bits.
    ///
    /// # Note
    ///
    /// This assumes that your texture compression format uses 4x4 blocks. This is true for all BC1-BC7 formats,
    /// but ASTC can use other block sizes.
    ///
    /// # Optional Parameters
    /// - `bits`: Defaults to `5`
    ///
    /// # Safety
    ///
    /// The Nvidia SDK does not declare what a safe value for `bits` may be. If it is too large or
    /// small, it will cause undefined behaviour. Assuming the operation has not changed much since the
    /// original [open sourced implementation](https://github.com/castano/nvidia-texture-tools/blob/master/src/nvtt/Surface.cpp),
    /// then `bits` cannot be larger than 32, as it is casted to a `c_int` and then performs the bitshift
    /// `1 << bits`. No current version lower/upper bounds are formally stated.
    ///
    /// Only the value `bits = 5`is mentioned, however a far broader range of values is likely safe.
    pub unsafe fn block_scale_cocg(&mut self, bits: Option<u32>) {
        let bits = bits.unwrap_or(5);
        // Ignored
        let threshold = 0.0;

        unsafe {
            nvtt_sys::nvttSurfaceBlockScaleCoCg(
                self.0,
                bits as i32,
                threshold,
                std::ptr::null_mut(),
            )
        }
    }

    /// Converts from `(Co, Cg, scale, Y)` colors to `(r, g, b, 1)` colors.
    ///
    /// This is useful for formats that use chroma subsampling.
    /// Y is in the range `[0, 1]`, while Co and Cg are in the range `[-1, 1]`. Co and Cg are
    /// multiplied by channel 2 (`scale`) to reverse the effects of optionally calling
    /// [`Surface::block_scale_cocg()`].
    ///
    /// The YCoCg-to-RGB formula used is
    ///
    /// ```text
    ///r = Y + Co - Cg
    /// g = Y + Cg
    /// b = Y - Co - Cg
    /// ```
    pub fn from_ycocg(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceToYCoCg(self.0, std::ptr::null_mut());
        }
    }

    /// Stores luminance-only values in a two-channel way. Maybe consider BC4 compression instead.
    ///
    /// Luminance `L` is computed by averaging the red, green, and blue values, while `M` stores
    /// the max of these values and `threshold`. The red, green, and blue channels then store `L/M`,
    /// and the alpha channel stores `(M - threshold)/(1 - threshold)`.
    ///
    /// # Optional Parameters
    /// - `range`: Defaults to `1.0`
    /// - `threshold`: Defaults to `0.0`
    pub fn to_lm(&mut self, range: Option<f32>, threshold: Option<f32>) {
        let range = range.unwrap_or(1.0);
        let threshold = threshold.unwrap_or(0.0);

        unsafe {
            nvtt_sys::nvttSurfaceToLM(self.0, range, threshold, std::ptr::null_mut());
        }
    }

    /// Converts from RGB colors to a (U, V, W, L) color space, much like RGBM.
    ///
    /// All values are clamped to `[0, 1]`. Then a luminance-like value `L` is computed from RGB using
    ///
    /// `L = max(sqrtf(R^2 + G^2 + B^2), 1e-6f)`.
    ///
    /// This then stores the value `(R/L, G/L, B/L, L/sqrt(3))`.
    ///
    /// # Optional Paramters
    /// - `range`: Defaults to `1.0`
    pub fn to_luvw(&mut self, range: Option<f32>) {
        let range = range.unwrap_or(1.0);

        unsafe {
            nvtt_sys::nvttSurfaceToLUVW(self.0, range, std::ptr::null_mut());
        }
    }

    /// Converts from [`Surface::to_luvw()`]'s color space to RGB colors.
    ///
    /// This is the same as [`Surface::from_rgbm()`] with `threshold = range * sqrt(3))`.
    ///
    /// # Optional Paramters
    /// - `range`: Defaults to `1.0`
    pub fn from_luvw(&mut self, range: Option<f32>) {
        let range = range.unwrap_or(1.0);

        unsafe {
            nvtt_sys::nvttSurfaceFromLUVW(self.0, range, std::ptr::null_mut());
        }
    }

    /// Replaces all values with their log with the given base.
    pub fn to_log_scale(&mut self, channel: Channel, base: f32) {
        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceToLogScale(self.0, channel, base, std::ptr::null_mut());
        }
    }

    /// Inverts [`Surface::to_log_scale()`] by replacing all values `x` with `base^x`.
    pub fn from_log_scale(&mut self, channel: Channel, base: f32) {
        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceFromLogScale(self.0, channel, base, std::ptr::null_mut());
        }
    }

    /// Returns the approximate fraction (0 to 1) of the image with an alpha value greater than `alpha_ref`.
    ///
    /// This function uses 8 x 8 subsampling together with linear interpolation.
    ///
    /// # Notes
    /// `alpha_ref` is clamped to the range `[1/256, 255/256]`.
    pub fn alpha_test_coverage(&self, alpha_ref: f32, alpha_channel: Channel) -> f32 {
        let alpha_channel = alpha_channel as i32;

        unsafe { nvtt_sys::nvttSurfaceAlphaTestCoverage(self.0, alpha_ref, alpha_channel) }
    }

    /// Attempts to scale the alpha channel so that a fraction `coverage` (between 0 and 1) of
    /// the surface has an alpha greater than `alpha_ref`.
    ///
    /// See also [`Surface::alpha_test_coverage()`] for the method used to determine what fraction passes the alpha test.
    pub fn scale_alpha_to_coverage(
        &mut self,
        coverage: f32,
        alpha_ref: f32,
        alpha_channel: Channel,
    ) {
        let alpha_channel = alpha_channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceScaleAlphaToCoverage(
                self.0,
                coverage,
                alpha_ref,
                alpha_channel,
                std::ptr::null_mut(),
            )
        }
    }

    /// Computes the average of a channel, possibly with alpha or with a gamma transfer function.
    ///
    /// If `alpha_channel` is `None`, this function computes
    ///
    /// `(sum(c[i]^gamma, i=0...numPixels)/numPixels)^(1/gamma)`
    ///
    /// where `c` is the channel's data.
    ///
    /// Otherwise, this computes
    ///
    /// `(sum((c[i]^gamma) * a[i], i=0...numPixels)/sum(a[i], i=0...numPixels))^(1/gamma)`
    ///
    /// where `a` is the alpha channel's data.
    ///
    /// # Optional Parameters
    /// - `gamma`: Defaults to `2.2`
    pub fn average(
        &self,
        channel: Channel,
        alpha_channel: Option<Channel>,
        gamma: Option<f32>,
    ) -> f32 {
        let gamma = gamma.unwrap_or(2.2);
        let channel = channel as i32;
        let alpha_channel = alpha_channel.map(|x| x as i32).unwrap_or(-1);

        unsafe { nvtt_sys::nvttSurfaceAverage(self.0, channel, alpha_channel, gamma) }
    }

    /// Stores a histogram of channel values between `range_min` and `range_max` into `bins`.
    ///
    /// This function does not clear `bins`' values, in case we want to accumulate multiple histograms.
    ///
    /// Each texel's value is linearly mapped to a bin, using floor rounding. Values below `range_min`
    /// are clamped to the first bin, values above `range_max` are clamped to the last bin. Then the bin's
    /// value is incremented.
    ///
    /// # Panics
    ///
    /// Panics if `bins.len() == 0`.
    pub fn histogram(&self, channel: Channel, range_min: f32, range_max: f32, bins: &mut [i32]) {
        if bins.is_empty() {
            panic!("bins must be non empty");
        }

        let channel = channel as i32;
        let bin_count = bins.len() as i32;
        let bin_ptr = bins.as_mut_ptr();

        unsafe {
            nvtt_sys::nvttSurfaceHistogram(
                self.0,
                channel,
                range_min,
                range_max,
                bin_count,
                bin_ptr,
                std::ptr::null_mut(),
            )
        }
    }

    /// Returns the minimum and maximum value in this channel, possibly using alpha testing.
    ///
    /// If `alpha_channel` is `None`, this returns the minimum and maximum value in this channel.
    ///
    /// Otherwise, this only includes texels for which the alpha value is greater than `alpha_ref`.
    ///
    /// If an alpha channel is selected and all texels fail the alpha test, this sets this will return
    /// [`f32::MAX`] and to [`f32::MIN`], i.e. for a return value `ret` one will have `ret.0 > ret.1`.
    pub fn range(
        &self,
        channel: Channel,
        alpha_channel: Option<Channel>,
        alpha_ref: f32,
    ) -> (f32, f32) {
        let channel = channel as i32;
        let alpha_channel = alpha_channel.map(|x| x as i32).unwrap_or(-1);

        // Will be overwritten
        let mut range_min: f32 = 0.0;
        let mut range_max: f32 = 0.0;
        let range_min_ptr: *mut f32 = &mut range_min;
        let range_max_ptr: *mut f32 = &mut range_max;

        unsafe {
            nvtt_sys::nvttSurfaceRange(
                self.0,
                channel,
                range_min_ptr,
                range_max_ptr,
                alpha_channel,
                alpha_ref,
                std::ptr::null_mut(),
            )
        }

        (range_min, range_max)
    }

    /// Applies a 4x4 affine transformation to the RGBA values.
    ///
    /// `w0...w3` are the columns of the matrix. `offset` is added after the matrix-vector multiplication.
    ///
    /// In other words, all `(r, g, b, a)` values are replaced with
    ///
    ///```text
    /// (r)   (w0[0], w1[0], w2[0], w3[0]) (r)   (offset[0])
    /// (g) = (w0[1], w1[1], w2[1], w3[1]) (g) + (offset[1])
    /// (b)   (w0[2], w1[2], w2[2], w3[2]) (b)   (offset[2])
    /// (a)   (w0[3], w1[3], w2[3], w3[3]) (a)   (offset[3])
    ///```
    pub fn transform(
        &mut self,
        w0: [f32; 4],
        w1: [f32; 4],
        w2: [f32; 4],
        w3: [f32; 4],
        offset: [f32; 4],
    ) {
        unsafe {
            nvtt_sys::nvttSurfaceTransform(
                self.0,
                w0.as_ptr(),
                w1.as_ptr(),
                w2.as_ptr(),
                w3.as_ptr(),
                offset.as_ptr(),
                std::ptr::null_mut(),
            );
        }
    }

    /// Swizzles the channels of the surface.
    ///
    /// Each argument specifies where the corresponding channel should come from.
    /// For instance, setting `r` to [`Swizzle::B`] would mean that the red channel
    /// would be set to the current blue channel.
    ///
    /// In addition, the special values [`Swizzle::One`], [`Swizzle::Zero`], and
    /// [`Swizzle::NegOne`] represent setting the channel to a constant value of
    /// `1.0`, `0.0`, or `-1.0`, respectively.
    pub fn swizzle(&mut self, r: Swizzle, g: Swizzle, b: Swizzle, a: Swizzle) {
        unsafe {
            nvtt_sys::nvttSurfaceSwizzle(
                self.0,
                r as i32,
                g as i32,
                b as i32,
                a as i32,
                std::ptr::null_mut(),
            )
        }
    }

    /// Applies a `scale` and `bias` to the given channel. Each value `x` is replaced by `x * scale + bias`.
    pub fn scale_bias(&mut self, channel: Channel, scale: f32, bias: f32) {
        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceScaleBias(self.0, channel, scale, bias, std::ptr::null_mut());
        }
    }

    /// Replaces all colors by their absolute value.
    pub fn abs(&mut self, channel: Channel) {
        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceAbs(self.0, channel, std::ptr::null_mut());
        }
    }

    /// Clamps all values in the channel to the range `[low, high]`.
    pub fn clamp(&mut self, channel: Channel, low: f32, high: f32) {
        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceClamp(self.0, channel, low, high, std::ptr::null_mut());
        }
    }

    /// Interpolates all texels between their current color and a constant color `(r, g, b, a)`.
    ///
    /// `t` is the value used for linearly interpolating between the surface's current colors
    /// and the constant color. For instance, a value of `t=0` has no effect to the surface's
    /// colors, and a value of `t=1` replaces the surface's colors entirely with `(r, g, b, a)`.
    pub fn blend(&mut self, rgba: [f32; 4], t: f32) {
        let r = rgba[0];
        let g = rgba[1];
        let b = rgba[2];
        let a = rgba[3];
        let t = t.clamp(0.0, 1.0);

        unsafe { nvtt_sys::nvttSurfaceBlend(self.0, r, g, b, a, t, std::ptr::null_mut()) }
    }

    /// Convolves a channel with a kernel.
    ///
    /// This uses a 2D `dim x dim` kernel, with values in `kernel` specified in row-major order.
    /// The behavior around image borders is determined by [`Surface::wrap_mode()`].
    ///
    /// # Panics
    ///
    /// Panics if `dim * dim > kernel.len()`
    pub fn convolve_slice(&mut self, channel: Channel, dim: u32, kernel: &mut [f32]) {
        if dim * dim > kernel.len() as u32 {
            panic!("kernel does not hold enough values");
        }

        if dim == 0 {
            panic!("kernel must not be empty");
        }

        let channel = channel as i32;
        let kernel_ptr = kernel.as_mut_ptr();

        unsafe {
            nvtt_sys::nvttSurfaceConvolve(
                self.0,
                channel,
                dim as i32,
                kernel_ptr,
                std::ptr::null_mut(),
            );
        }
    }

    /// Similar to [`Surface::convolve_slice()`], this convolves a channel with a kernel.
    ///
    /// Values should be specified in row-major order.
    pub fn convolve<const N: usize>(&mut self, channel: Channel, mut kernel: [[f32; N]; N]) {
        if N == 0 {
            panic!("kernel must not be empty");
        }

        let channel = channel as i32;
        let kernel_ptr = kernel[0].as_mut_ptr();

        unsafe {
            nvtt_sys::nvttSurfaceConvolve(
                self.0,
                channel,
                N as i32,
                kernel_ptr,
                std::ptr::null_mut(),
            );
        }
    }

    /// Sets values in the given channel to either 1 or 0 depending on if they're greater
    /// than the `threshold`, with optional `dithering`.
    ///
    /// If dither is true, this uses Floyd-Steinberg dithering on the CPU. Not supported for 3D surfaces.
    ///
    /// # Panics
    ///
    /// Panics if `dither` is true and `depth > 1`
    pub fn binarize(&mut self, channel: Channel, threshold: f32, dither: bool) {
        if dither && self.depth() > 1 {
            panic!("binarize dithering not supported for 3D surfaces");
        }
        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceBinarize(
                self.0,
                channel,
                threshold,
                dither.into(),
                std::ptr::null_mut(),
            );
        }
    }

    /// Quantizes this channel to a particular number of `bits`, with optional `dithering`.
    /// Assumes input is in the `[0, 1]` range. Output  is in the `[0, 1]` range, but rounded to the
    /// middle of each bin.
    ///
    /// # Parameters
    /// - `channel`: The index of the channel to quantize.
    /// - `bits`: The number of bits to quantize to, yielding `2^bits` possible values.
    /// - `exact_endpoints`: If true, the set of quantized values will be `0, 1/(2^bits-1), ..., 1`,
    /// rather than `0, 1/(2^bits), ..., (2^bits-1)/(2^bits)`.
    /// - `dither`: If true, uses Floyd-Steinberg dithering on the CPU. Not supported for 3D surfaces.
    ///
    /// # Panics
    ///
    /// Panics if `dither` is true and `depth > 1`
    ///
    /// # Safety
    ///
    /// The Nvidia SDK does not declare what a safe value for `bits` may be. If it is too large or
    /// small, it will cause undefined behaviour. Assuming the operation has not changed much since the
    /// original [open sourced implementation](https://github.com/castano/nvidia-texture-tools/blob/master/src/nvtt/Surface.cpp),
    /// then `bits` cannot be larger than 32, as it is casted to a `c_int` and then performs the bitshift
    /// `1 << bits`. No current version lower/upper bounds are formally stated.
    pub unsafe fn quantize(
        &mut self,
        channel: Channel,
        bits: u32,
        exact_endpoints: bool,
        dither: bool,
    ) {
        if dither && self.depth() > 1 {
            panic!("quantize dithering not supported for 3D surfaces");
        }

        let channel = channel as i32;

        unsafe {
            nvtt_sys::nvttSurfaceQuantize(
                self.0,
                channel,
                bits as i32,
                exact_endpoints.into(),
                dither.into(),
                std::ptr::null_mut(),
            );
        }
    }

    /// Sets the RGB channels to a normal map generated by interpreting the alpha channel as a heightmap,
    /// using a blend of four small-scale to large-scale Sobel kernels.
    ///
    /// This uses a 9x9 kernel which is a weighted sum of a 3x3 (small), 5x5 (medium), 7x7 (big), and 9x9
    /// (large) differentiation kernels. Each of the weights can be greater than 1, or even negative.
    /// However, the kernel will be normalized so that its elements sum to 1, so scaling should be done on
    /// the alpha channel beforehand. The smallest kernel focuses on the highest-frequency details, and
    /// larger kernels attenuate higher frequencies.
    ///
    /// The source alpha channel, which is used as a height map to differentiate, is copied to the output.
    ///
    /// The output RGB channels will be in the ranges `[-1, 1]`, `[-1, 1]`, and `[0, 1]`.
    pub fn to_normal_map(&mut self, sm: f32, medium: f32, big: f32, large: f32) {
        unsafe {
            nvtt_sys::nvttSurfaceToNormalMap(self.0, sm, medium, big, large, std::ptr::null_mut());
        }
    }

    /// Applies a 3D->2D normal transformation, setting the `z` (blue) channel to `0`.
    pub fn transform_normals(&mut self, transform: NormalTransform) {
        unsafe {
            nvtt_sys::nvttSurfaceTransformNormals(self.0, transform.into(), std::ptr::null_mut());
        }
    }

    /// Reconstructs 3D normals from 2D transformed normals.
    pub fn reconstruct_normals(&mut self, transform: NormalTransform) {
        unsafe {
            nvtt_sys::nvttSurfaceReconstructNormals(self.0, transform.into(), std::ptr::null_mut());
        }
    }

    /// Sets the `z` (blue) channel to `x^2 + y^2`.
    ///
    /// If the x and y channels represent slopes, instead of normals, then this represents a
    /// CLEAN map. The important thing about this is that it can be mipmapped, and the difference
    /// between the sum of the square of the first and second mipmapped channels and the third
    /// mipmapped channel can be used to determine how rough the normal map is in a given area.
    ///
    /// This is a lower-memory and lower-bandwidth version of LEAN mapping, but it has the
    /// drawback that it can only represent isotropic roughness.
    pub fn to_clean_normal_map(&mut self) {
        unsafe {
            nvtt_sys::nvttSurfaceToCleanNormalMap(self.0, std::ptr::null_mut());
        }
    }

    /// Packs signed normals in `[-1, 1]` to an unsigned range `[0, 1]`, using
    ///
    /// `(r, g, b, a) |-> (r/2+1/2, g/2+1/2, b/2+1/2, a)`.
    pub fn pack_normals(&mut self) {
        let scale = 0.5;
        let bias = 0.5;

        unsafe {
            nvtt_sys::nvttSurfacePackNormals(self.0, scale, bias, std::ptr::null_mut());
        }
    }

    /// Expands packed normals in `[0, 1]` to signed normals in `[-1, 1]`, using
    ///
    /// `(r, g, b, a) |-> (2r-1, 2g-1, 2b-1, a)`.
    pub fn unpack_normals(&mut self) {
        let scale = 2.0;
        let bias = -1.0;

        unsafe {
            nvtt_sys::nvttSurfaceExpandNormals(self.0, scale, bias, std::ptr::null_mut());
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            nvtt_sys::nvttDestroySurface(self.0);
        }
    }
}

impl Clone for Surface {
    fn clone(&self) -> Self {
        unsafe {
            let ptr = nvtt_sys::nvttSurfaceClone(self.0);
            if ptr.is_null() {
                panic!("failed to allocate");
            } else {
                Self(ptr)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use crate::CUDA_SUPPORTED;
    use crate::{Channel, InputFormat, Surface, TextureType};
    use approx::assert_relative_eq;

    #[test]
    fn pixel() {
        let input_format = InputFormat::Bgra8Ub {
            data: &[255, 0, 0, 0],
            unsigned_to_signed: false,
        };
        let surface = Surface::image(input_format, 1, 1, 1).unwrap();

        assert_eq!(1, surface.width());
        assert_eq!(1, surface.height());
        assert_eq!(1, surface.depth());

        assert_eq!(TextureType::D2, surface.tex_type());

        assert_eq!(4, surface.data().len());
        assert_eq!(1, surface.channel(Channel::R).len());
        assert_eq!(1, surface.channel(Channel::G).len());
        assert_eq!(1, surface.channel(Channel::B).len());
        assert_eq!(1, surface.channel(Channel::A).len());

        assert_relative_eq!(0.0, surface.data()[0]);
        assert_relative_eq!(0.0, surface.data()[1]);
        assert_relative_eq!(1.0, surface.data()[2]);
        assert_relative_eq!(0.0, surface.data()[3]);

        assert_relative_eq!(0.0, surface.channel(Channel::R)[0]);
        assert_relative_eq!(0.0, surface.channel(Channel::G)[0]);
        assert_relative_eq!(1.0, surface.channel(Channel::B)[0]);
        assert_relative_eq!(0.0, surface.channel(Channel::A)[0]);
    }

    // According to docs
    // If unsignedToSigned is true, InputFormat_BGRA_8UB unsigned input will be converted to
    // signed values between -1 and 1, mapping 0 to -1, and 1...255 linearly to -1...1.
    fn unsigned_conv(x: u8) -> f32 {
        match x {
            0 => -1.0,
            x => {
                // [1, 255] => [0, 254]
                let x = (x - 1) as f32;
                let t = x / 254.0;

                -1.0 + t * (2.0)
            }
        }
    }

    #[test]
    fn unsigned_to_signed() {
        let b = 1;
        let g = 0;
        let r = 128;
        let a = 255;

        let input_format = InputFormat::Bgra8Ub {
            data: &[b, g, r, a],
            unsigned_to_signed: true,
        };
        let surface = Surface::image(input_format, 1, 1, 1).unwrap();

        assert_relative_eq!(unsigned_conv(r), surface.channel(Channel::R)[0]);
        assert_relative_eq!(unsigned_conv(g), surface.channel(Channel::G)[0]);
        assert_relative_eq!(unsigned_conv(b), surface.channel(Channel::B)[0]);
        assert_relative_eq!(unsigned_conv(a), surface.channel(Channel::A)[0]);

        let g_conv = surface.channel(Channel::G)[0];
        let b_conv = surface.channel(Channel::B)[0];
        assert_relative_eq!(g_conv, b_conv);

        assert_relative_eq!(0.0, surface.channel(Channel::R)[0]);
        assert_relative_eq!(-1.0, surface.channel(Channel::G)[0]);
        assert_relative_eq!(-1.0, surface.channel(Channel::B)[0]);
        assert_relative_eq!(1.0, surface.channel(Channel::A)[0]);
    }

    #[cfg(feature = "cuda")]
    const BASIC_INPUT: InputFormat = InputFormat::Bgra8Ub {
        data: &[255, 255, 255, 255],
        unsigned_to_signed: false,
    };

    // Test cpu gpu behaviour with channel / data accesses
    // channel() / data()         should not force a CPU copy
    // channel_mut() / data_mut() should force a CPU copy
    #[test]
    #[cfg(feature = "cuda")]
    fn channel_mut_cpu_gpu() {
        if *CUDA_SUPPORTED {
            let mut surface = Surface::image(BASIC_INPUT, 1, 1, 1).unwrap();
            // Not on gpu by default
            assert!(!surface.on_gpu());
            assert!(surface.gpu_data_ptr().is_null());

            // Upload to gpu
            surface.to_gpu();
            assert!(surface.on_gpu());
            assert!(!surface.gpu_data_ptr().is_null());

            // Still on gpu
            assert_relative_eq!(1.0, surface.channel(Channel::R)[0]);
            assert!(surface.on_gpu());
            assert!(!surface.gpu_data_ptr().is_null());

            // Not on gpu after mutable access
            surface.channel_mut(Channel::R)[0] = 0.0;
            assert!(!surface.on_gpu());
            assert_relative_eq!(0.0, surface.channel(Channel::R)[0]);

            // Reupload to gpu
            surface.to_gpu();
            assert!(surface.on_gpu());
            assert!(!surface.gpu_data_ptr().is_null());

            // Stay on gpu (immutable access)
            assert_relative_eq!(0.0, surface.data()[0]);
            assert!(surface.on_gpu());
            assert!(!surface.gpu_data_ptr().is_null());

            // Remove from gpu
            surface.data_mut()[0] = 1.0;
            assert!(!surface.on_gpu());
            assert_relative_eq!(1.0, surface.channel(Channel::R)[0]);

            // Finally upload back as final test
            surface.to_gpu();
            assert!(surface.on_gpu());
            assert!(!surface.gpu_data_ptr().is_null());
        }
    }

    // Test cpu gpu behaviour with library function (should not require a CPU copy)
    #[test]
    #[cfg(feature = "cuda")]
    fn function_mut_cpu_gpu() {
        if *CUDA_SUPPORTED {
            let mut surface = Surface::image(BASIC_INPUT, 1, 1, 1).unwrap();

            surface.to_gpu();
            assert!(surface.on_gpu());

            // Immutable access should still keep it on GPU
            let old_r = surface.channel(Channel::R)[0];
            let old_g = surface.channel(Channel::G)[0];
            let old_b = surface.channel(Channel::B)[0];
            let old_a = surface.channel(Channel::A)[0];
            assert!(surface.on_gpu());

            let w0 = [2., 0., 0., 0.];
            let w1 = [0., 4., 0., 0.];
            let w2 = [0., 0., 6., 0.];
            let w3 = [0., 0., 0., 8.];
            let offset = [1., 1., 1., 1.];
            surface.transform(w0, w1, w2, w3, offset);
            assert!(surface.on_gpu());

            // Will incur a copy, but should still be on GPU
            let new_r = surface.channel(Channel::R)[0];
            assert!(surface.on_gpu());
            assert_relative_eq!(2.0 * old_r + 1.0, new_r);
            assert_relative_eq!(4.0 * old_g + 1.0, surface.channel(Channel::G)[0]);
            assert_relative_eq!(6.0 * old_b + 1.0, surface.channel(Channel::B)[0]);
            assert_relative_eq!(8.0 * old_a + 1.0, surface.channel(Channel::A)[0]);
        }
    }

    #[test]
    fn rgbe() {
        let input_format = InputFormat::Bgra8Ub {
            data: &[32, 64, 128, 234, 255, 32, 64, 85],
            unsigned_to_signed: false,
        };

        let mut surface = Surface::image(input_format, 2, 1, 1).unwrap();
        unsafe {
            surface.to_rgbe(25, 25);
            surface.from_rgbe(25, 25);
        }
    }

    #[test]
    fn compression() {
        use crate::{
            CompressionOptions, Context, Format, InputFormat, OutputOptions, Quality, Surface,
        };
        let input = InputFormat::Bgra8Ub {
            data: &[0u8; 16 * 16 * 4],
            unsigned_to_signed: false,
        };
        let image = Surface::image(input, 16, 16, 1).unwrap();

        let context = Context::new();
        // Hacky but removes warnings for now so whatever (it's a test)
        #[cfg(feature = "cuda")]
        let context = {
            if *crate::CUDA_SUPPORTED {
                let mut context = context;
                context.set_cuda_acceleration(true);
                context
            } else {
                context
            }
        };

        let mut compression_options = CompressionOptions::new();
        compression_options.set_quality(Quality::Fastest);

        let output_options = OutputOptions::new();

        compression_options.set_format(Format::Bc1);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16 / 2, bytes.len());

        compression_options.set_format(Format::Bc2);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16, bytes.len());

        compression_options.set_format(Format::Bc3);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16, bytes.len());

        compression_options.set_format(Format::Bc4S);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16 / 2, bytes.len());

        compression_options.set_format(Format::Bc5S);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16, bytes.len());

        compression_options.set_format(Format::Bc6S);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16, bytes.len());

        compression_options.set_format(Format::Bc7);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16, bytes.len());

        compression_options.set_format(Format::Rgba);
        let bytes = context
            .compress(&image, &compression_options, &output_options)
            .unwrap();
        assert_eq!(16 * 16 * 4, bytes.len());
    }
}
