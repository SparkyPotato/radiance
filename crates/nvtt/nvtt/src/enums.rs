#[allow(unused_imports)]
use crate::{CubeSurface, Surface};

/// Represents an RGBA channel. For various operations with [`Surface`] and [`CubeSurface`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Channel {
    /// Red channel, referred to as channel 0 in the C/C++ API
    R = 0,
    /// Green channel, referred to as channel 1 in the C/C++ API
    G = 1,
    /// Blue channel, referred to as channel 2 in the C/C++ API
    B = 2,
    /// Alpha channel, referred to as channel 3 in the C/C++ API
    A = 3,
}

/// Container type for encoded data.
///
/// # Notes
///
/// For DDS containers, NVTT stores some additional data in the `reserved[]` fields to allow consumers to detect writer versions.
///
/// - `reserved[7]` is the FourCC code "UVER", and `reserved[8]` stores a version number that can be set by the user.
/// - `reserved[9]` is the FourCC code "NVTT", and `reserved[10]` is the NVTT writer version (which isn't necessarily the same as [`version()`](crate::version)).
///
/// For DDS containers, NVTT also extends the `dwFlags` field with two more flags.
/// - `DDPF_SRGB (0x40000000U)` indicates that the texture uses an sRGB transfer function. Note that most readers will ignore this and instead guess the transfer function from the format.
/// - `DDPF_NORMAL (0x80000000U)` indicates that the texture is a normal map.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Container {
    /// DDS without the DX10 header extension. Compatible with legacy readers, but doesn't support BC6 or BC7.
    Dds,
    /// DDS without the DX10 header. Supports BC6 and BC7, but may be unreadable by legacy readers.
    Dds10,
}

use nvtt_sys::NvttContainer;
impl From<Container> for NvttContainer {
    fn from(val: Container) -> Self {
        match val {
            Container::Dds => NvttContainer::NVTT_Container_DDS,
            Container::Dds10 => NvttContainer::NVTT_Container_DDS10,
        }
    }
}

/// Affects how certain cube surface processing algorithms work. For use with
/// [`CubeSurface::cosine_power_filter()`] and [`CubeSurface::fast_resample()`]
///
/// Use [`EdgeFixup::None`] if unsure.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum EdgeFixup {
    /// No effect.
    None,
    /// Slightly stretches and shifts the coordinate systems [`CubeSurface::cosine_power_filter()`]
    /// and [`CubeSurface::fast_resample()`] use.
    Stretch,
    /// Applies a cubic warp to each face's coordinate system in [`CubeSurface::cosine_power_filter()`]
    /// and [`CubeSurface::fast_resample()`], warping texels closer to edges more.
    Warp,
}

impl From<EdgeFixup> for nvtt_sys::EdgeFixup {
    fn from(val: EdgeFixup) -> Self {
        match val {
            EdgeFixup::None => nvtt_sys::EdgeFixup::NVTT_EdgeFixup_None,
            EdgeFixup::Stretch => nvtt_sys::EdgeFixup::NVTT_EdgeFixup_Stretch,
            EdgeFixup::Warp => nvtt_sys::EdgeFixup::NVTT_EdgeFixup_Warp,
        }
    }
}

/// Quality modes.
///
/// These can be used to trade off speed of compression for lower error, and often selects the specific compression algorithm that will be used. Here's a table showing which (format, quality) combinations support CUDA acceleration:
///
/// | Quality     | BC1 | BC1a | BC2 | BC3 | BC3n | RGBM | BC4 | BC5 | BC6 | BC7 | ASTC       |
/// |-------------|-----|------|-----|-----|------|------|-----|-----|-----|-----|------------|
/// | Fastest     | Yes | No   | No  | No  | No   | No   | Yes | Yes | Yes | Yes | Yes        |
/// | Normal      | Yes | Yes  | Yes | Yes | No   | No   | Yes | Yes | Yes | Yes | Yes        |
/// | Production  | Yes | Yes  | Yes | Yes | No   | No   | No  | No  | No  | No  | Yes (slow) |
/// | Highest     | Yes | Yes  | Yes | Yes | No   | No   | No  | No  | No  | No  | Yes (slow) |
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Quality {
    Fastest,
    Normal,
    Production,
    Highest,
}

use nvtt_sys::NvttQuality;
impl From<Quality> for NvttQuality {
    fn from(val: Quality) -> Self {
        match val {
            Quality::Fastest => NvttQuality::NVTT_Quality_Fastest,
            Quality::Normal => NvttQuality::NVTT_Quality_Normal,
            Quality::Production => NvttQuality::NVTT_Quality_Production,
            Quality::Highest => NvttQuality::NVTT_Quality_Highest,
        }
    }
}

/// Alpha mode. For use with [`Surface::set_alpha_mode()`]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AlphaMode {
    /// This image has no alpha. The alpha channel will be ignored in some forms of compression.
    None,
    /// Alpha represents opacity; for instance, `(r, g, b, 0.5)` is a 50% opaque `(r, g, b)` color.
    Transparency,
    /// Colors are stored using premultiplied alpha: `(a*r, a*g, a*b, a)` is an `(r, g, b)` color
    /// with an opacity of `a`. This is mostly for tracking purposes; compressors only distinguish
    /// between [`None`](AlphaMode::None) and [`Transparency`](AlphaMode::Transparency).
    Premultiplied,
}

use nvtt_sys::NvttAlphaMode;
impl From<AlphaMode> for NvttAlphaMode {
    fn from(val: AlphaMode) -> Self {
        match val {
            AlphaMode::None => NvttAlphaMode::NVTT_AlphaMode_None,
            AlphaMode::Transparency => NvttAlphaMode::NVTT_AlphaMode_Transparency,
            AlphaMode::Premultiplied => NvttAlphaMode::NVTT_AlphaMode_Premultiplied,
        }
    }
}

impl From<NvttAlphaMode> for AlphaMode {
    fn from(other: NvttAlphaMode) -> Self {
        match other {
            NvttAlphaMode::NVTT_AlphaMode_None => Self::None,
            NvttAlphaMode::NVTT_AlphaMode_Transparency => Self::Transparency,
            NvttAlphaMode::NVTT_AlphaMode_Premultiplied => Self::Premultiplied,
        }
    }
}

/// Supported block-compression formats, including compressor variants.
///
/// That is:
///
/// - 'DXT1' is a format, 'DXT1a' and 'DXT1n' are DXT1 compressors.
/// - 'DXT3' is a format, 'DXT3n' is a DXT3 compressor.
#[derive(Clone, Copy, Debug)]
pub enum Format {
    /// Linear RGB format.
    Rgb,
    /// Linear RGBA format. Same as [`Format::Rgb`].
    Rgba, // = Rgb
    /// DX9 - DXT1 format.
    Dxt1,
    /// DX9 - DXT1 with binary alpha.
    Dxt1a,
    /// DX9 - DXT3 format.
    Dxt3,
    /// DX9 - DXT5 format.
    Dxt5,
    /// DX9 - DXT5 normal format.
    /// Stores a normal `(x, y, z)` as `(R, G, B, A) = (1, y, 0, x)`.
    Dxt5n,
    /// DX10 - BC1 (DXT1) format. Same as [`Format::Dxt1`].
    Bc1, // = NVTT_Format_DXT1
    /// DX10 - BC1 (DXT1) format. Same as [`Format::Dxt1a`].
    Bc1a, // = NVTT_Format_DXT1a
    /// DX10 - BC2 (DXT3) format. Same as [`Format::Dxt3`].
    Bc2, // = NVTT_Format_DXT3
    /// DX10 - BC3 (DXT5) format. Same as [`Format::Dxt5`].
    Bc3, // = NVTT_Format_DXT5
    /// DX10 - BC3 (DXT5) normal format.
    /// Stores a normal `(x, y, z)` as `(1, y, 0, x)`.
    ///
    /// Same as [`Format::Dxt5n`].
    Bc3n, // = NVTT_Format_DXT5n
    /// DX10 - BC4U (ATI1) format (one channel, unsigned).
    Bc4,
    /// DX10 - BC4S format (one channel, signed).
    Bc4S,
    /// DX10 - ATI2 format, similar to BC5U, channel order GR instead of RG.
    Ati2,
    /// DX10 - BC5U format (two channels, unsigned).
    Bc5,
    /// DX10 - BC5S format (two channels, signed).
    Bc5S,
    /// DX10 - BC6 format (three-channel HDR, unsigned).
    Bc6U,
    /// DX10 - BC6 format (three-channel HDR, signed).
    Bc6S,
    /// DX10 - BC7 format (four channels, UNORM).
    Bc7,
    /// DX10 - BC3(DXT5) - using a magnitude encoding to approximate three-channel
    /// HDR data in four UNORM channels. The input should be in the range `[0, 1]`,
    /// and this should give more accurate values closer to `0`. On most devices,
    /// consider using BC6 instead.
    ///
    /// To decompress this format, decompress it like a standard BC3 texture,
    /// then compute `(R, G, B)` from `(r, g, b, m)` using [`Surface::from_rgbm`] with `range = 1` and `threshold = 0.25`:
    ///
    /// `M = m * 0.75 + 0.25`
    ///
    /// `(R, G, B)` = `(r, g, b) * M`
    ///
    /// The idea is that since BC3 uses separate compression for the RGB and alpha blocks,
    /// the RGB and M signals can be independent. Additionally, the compressor can account
    /// for the RGB compression error. This will print warnings if any of the computed `m` values
    /// were greater than `1.0`.
    Bc3Rgbm,
    /// ASTC - LDR - format, tile size 4x4.
    AstcLdr4x4,
    /// ASTC - LDR - format, tile size 5x4.
    AstcLdr5x4,
    /// ASTC - LDR - format, tile size 5x5.
    AstcLdr5x5,
    /// ASTC - LDR - format, tile size 6x5.
    AstcLdr6x5,
    /// ASTC - LDR - format, tile size 6x6.
    AstcLdr6x6,
    /// ASTC - LDR - format, tile size 8x5.
    AstcLdr8x5,
    /// ASTC - LDR - format, tile size 8x6.
    AstcLdr8x6,
    /// ASTC - LDR - format, tile size 8x8.
    AstcLdr8x8,
    /// ASTC - LDR - format, tile size 10x5.
    AstcLdr10x5,
    /// ASTC - LDR - format, tile size 10x6.
    AstcLdr10x6,
    /// ASTC - LDR - format, tile size 10x8.
    AstcLdr10x8,
    /// ASTC - LDR - format, tile size 10x10.
    AstcLdr10x10,
    /// ASTC - LDR - format, tile size 12x10.
    AstcLdr12x10,
    /// ASTC - LDR - format, tile size 12x12.
    AstcLdr12x12,
}

use nvtt_sys::NvttFormat;
impl From<Format> for NvttFormat {
    fn from(val: Format) -> Self {
        match val {
            Format::Rgb => NvttFormat::NVTT_Format_RGB,
            Format::Rgba => NvttFormat::NVTT_Format_RGB, // = NVTT_Format_RGB
            Format::Dxt1 => NvttFormat::NVTT_Format_DXT1,
            Format::Dxt1a => NvttFormat::NVTT_Format_DXT1a,
            Format::Dxt3 => NvttFormat::NVTT_Format_DXT3,
            Format::Dxt5 => NvttFormat::NVTT_Format_DXT5,
            Format::Dxt5n => NvttFormat::NVTT_Format_DXT5n,
            Format::Bc1 => NvttFormat::NVTT_Format_DXT1, // = NVTT_Format_DXT1
            Format::Bc1a => NvttFormat::NVTT_Format_DXT1a, // = NVTT_Format_DXT1a
            Format::Bc2 => NvttFormat::NVTT_Format_DXT3, // = NVTT_Format_DXT3
            Format::Bc3 => NvttFormat::NVTT_Format_DXT5, // = NVTT_Format_DXT5
            Format::Bc3n => NvttFormat::NVTT_Format_DXT5n, // = NVTT_Format_DXT5n
            Format::Bc4 => NvttFormat::NVTT_Format_BC4,
            Format::Bc4S => NvttFormat::NVTT_Format_BC4S,
            Format::Ati2 => NvttFormat::NVTT_Format_ATI2,
            Format::Bc5 => NvttFormat::NVTT_Format_BC5,
            Format::Bc5S => NvttFormat::NVTT_Format_BC5S,
            Format::Bc6U => NvttFormat::NVTT_Format_BC6U,
            Format::Bc6S => NvttFormat::NVTT_Format_BC6S,
            Format::Bc7 => NvttFormat::NVTT_Format_BC7,
            Format::Bc3Rgbm => NvttFormat::NVTT_Format_BC3_RGBM,
            Format::AstcLdr4x4 => NvttFormat::NVTT_Format_ASTC_LDR_4x4,
            Format::AstcLdr5x4 => NvttFormat::NVTT_Format_ASTC_LDR_5x4,
            Format::AstcLdr5x5 => NvttFormat::NVTT_Format_ASTC_LDR_5x5,
            Format::AstcLdr6x5 => NvttFormat::NVTT_Format_ASTC_LDR_6x5,
            Format::AstcLdr6x6 => NvttFormat::NVTT_Format_ASTC_LDR_6x6,
            Format::AstcLdr8x5 => NvttFormat::NVTT_Format_ASTC_LDR_8x5,
            Format::AstcLdr8x6 => NvttFormat::NVTT_Format_ASTC_LDR_8x6,
            Format::AstcLdr8x8 => NvttFormat::NVTT_Format_ASTC_LDR_8x8,
            Format::AstcLdr10x5 => NvttFormat::NVTT_Format_ASTC_LDR_10x5,
            Format::AstcLdr10x6 => NvttFormat::NVTT_Format_ASTC_LDR_10x6,
            Format::AstcLdr10x8 => NvttFormat::NVTT_Format_ASTC_LDR_10x8,
            Format::AstcLdr10x10 => NvttFormat::NVTT_Format_ASTC_LDR_10x10,
            Format::AstcLdr12x10 => NvttFormat::NVTT_Format_ASTC_LDR_12x10,
            Format::AstcLdr12x12 => NvttFormat::NVTT_Format_ASTC_LDR_12x12,
        }
    }
}

/// Input formats. For use with [`Surface::image()`].
#[derive(Clone, Copy, Debug)]
pub enum InputFormat<'a> {
    /// Blue, green, red, and alpha channels. Each component is a `u8`, which is mapped to
    /// `[0, 1]`.
    Bgra8Ub {
        data: &'a [u8],
        /// If true, input will be converted to signed values between `-1` and `1`, mapping `0` to `-1`, and `1...255` linearly to `-1...1`.
        unsigned_to_signed: bool,
    },
    /// Blue, green, red, and alpha channels. Each component is an `i8`, which is mapped to
    /// `[-1, 1]`.
    Bgra8Sb(&'a [u8]),
    /// Red, green, blue, and alpha channels. Each component is an `f16` in native endianness.
    Rgba16f(&'a [u8]),
    /// Red, green, blue, and alpha channels. Each component is an `f32` in native endianness.
    Rgba32f(&'a [u8]),
    /// Red channel. Each value is an `f32` in native endianness.
    R32f(&'a [u8]),
}

/// Split input formats. For use with [`Surface::image_split()`].
#[derive(Clone, Copy, Debug)]
pub enum SplitInputFormat<'a> {
    /// Split blue, green, red, and alpha channels. Each component is a `u8`, which is mapped to
    /// `[0, 1]`.
    Bgra8Ub {
        b: &'a [u8],
        g: &'a [u8],
        r: &'a [u8],
        a: &'a [u8],
    },
    /// Split blue, green, red, and alpha channels. Each component is an `i8`, which is mapped to
    /// `[-1, 1]`.
    Bgra8Sb {
        b: &'a [u8],
        g: &'a [u8],
        r: &'a [u8],
        a: &'a [u8],
    },
    /// Split red, green, blue, and alpha channels. Each component is an `f16` in native endianness.
    Rgba16f {
        r: &'a [u8],
        g: &'a [u8],
        b: &'a [u8],
        a: &'a [u8],
    },
    /// Split red, green, blue, and alpha channels. Each component is an `f32` in native endianness.
    Rgba32f {
        r: &'a [u8],
        g: &'a [u8],
        b: &'a [u8],
        a: &'a [u8],
    },
    /// Single red channel. Each value is an `f32` in native endianness.
    R32f(&'a [u8]),
}

macro_rules! impl_input_format {
    ($input:ident) => {
        impl<'a> $input<'a> {
            /// Dimensionality of the uncompressed color format.
            pub fn dim(&self) -> usize {
                match self {
                    &Self::Bgra8Ub { .. } => 4,
                    &Self::Bgra8Sb { .. } => 4,
                    &Self::Rgba16f { .. } => 4,
                    &Self::Rgba32f { .. } => 1,
                    &Self::R32f { .. } => 1,
                }
            }

            /// Width in bytes per pixel per color channel of the uncompressed color format.
            pub fn width(&self) -> usize {
                match self {
                    &Self::Bgra8Ub { .. } => 1,
                    &Self::Bgra8Sb { .. } => 1,
                    &Self::Rgba16f { .. } => 2,
                    &Self::Rgba32f { .. } => 4,
                    &Self::R32f { .. } => 4,
                }
            }

            pub(crate) fn into_nvtt(self) -> nvtt_sys::NvttInputFormat {
                match self {
                    Self::Bgra8Ub { .. } => nvtt_sys::NvttInputFormat::NVTT_InputFormat_BGRA_8UB,
                    Self::Bgra8Sb { .. } => nvtt_sys::NvttInputFormat::NVTT_InputFormat_BGRA_8SB,
                    Self::Rgba16f { .. } => nvtt_sys::NvttInputFormat::NVTT_InputFormat_RGBA_16F,
                    Self::Rgba32f { .. } => nvtt_sys::NvttInputFormat::NVTT_InputFormat_RGBA_32F,
                    Self::R32f { .. } => nvtt_sys::NvttInputFormat::NVTT_InputFormat_R_32F,
                }
            }
        }
    };
}

impl_input_format!(InputFormat);
impl_input_format!(SplitInputFormat);

impl<'a> InputFormat<'a> {
    pub fn data(&self) -> &[u8] {
        match self {
            Self::Bgra8Ub { data, .. } => data,
            Self::Bgra8Sb(data) => data,
            Self::Rgba16f(data) => data,
            Self::Rgba32f(data) => data,
            Self::R32f(data) => data,
        }
    }

    pub(crate) fn min_bytes(&self, w: u32, h: u32, d: u32) -> u32 {
        let pixel_count = (w * h * d) as usize;
        let bytes_per_pixel = self.dim() * self.width();
        let byte_count = pixel_count * bytes_per_pixel;
        byte_count as u32
    }

    pub(crate) fn fit_dim(&self, w: u32, h: u32, d: u32) -> bool {
        self.data().len() >= self.min_bytes(w, h, d) as usize
    }
}

impl<'a> SplitInputFormat<'a> {
    pub(crate) fn min_bytes(&self, w: u32, h: u32, d: u32) -> u32 {
        let pixel_count = (w * h * d) as usize;
        let bytes_per_pixel_per_channel = self.width();
        let bytes_per_channel = pixel_count * bytes_per_pixel_per_channel;
        bytes_per_channel as u32
    }

    pub(crate) fn shortest_slice_len(&self) -> u32 {
        let mut lengths = [usize::MAX; 4];

        match self {
            Self::Bgra8Ub { b, g, r, a } => {
                lengths[0] = b.len();
                lengths[1] = g.len();
                lengths[2] = r.len();
                lengths[3] = a.len();
            }
            Self::Bgra8Sb { b, g, r, a } => {
                lengths[0] = b.len();
                lengths[1] = g.len();
                lengths[2] = r.len();
                lengths[3] = a.len();
            }
            Self::Rgba32f { r, g, b, a } => {
                lengths[0] = r.len();
                lengths[1] = g.len();
                lengths[2] = b.len();
                lengths[3] = a.len();
            }
            Self::Rgba16f { r, g, b, a } => {
                lengths[0] = r.len();
                lengths[1] = g.len();
                lengths[2] = b.len();
                lengths[3] = a.len();
            }
            Self::R32f(r) => lengths[0] = r.len(),
        }

        lengths.into_iter().min().unwrap() as u32
    }

    pub(crate) fn fit_dim(&self, w: u32, h: u32, d: u32) -> bool {
        self.shortest_slice_len() >= self.min_bytes(w, h, d)
    }
}

/// Pixel value types.
///
/// These are used for [`Format::Rgb`]: they indicate how the output should be interpreted, but do not have any influence over the input. They are ignored for other compression modes.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum PixelType {
    /// Used to indicate a `DXGI_..._UNORM` format
    UnsignedNorm,
    /// Used to indicate a `DXGI_..._FLOAT` format.
    Float,
    /// Used to indicate a `DXGI_..._UF16` format. Unused.
    UnsignedFloat,
    /// Shared exponent. Only supported for `DXGI_FORMAT_R9G9B9E5_SHAREDEXP.`
    SharedExp,
}

use nvtt_sys::NvttPixelType;
impl From<PixelType> for NvttPixelType {
    fn from(val: PixelType) -> Self {
        match val {
            PixelType::UnsignedNorm => NvttPixelType::NVTT_PixelType_UnsignedNorm,
            PixelType::Float => NvttPixelType::NVTT_PixelType_Float,
            PixelType::UnsignedFloat => NvttPixelType::NVTT_PixelType_UnsignedFloat,
            PixelType::SharedExp => NvttPixelType::NVTT_PixelType_SharedExp,
        }
    }
}

/// Texture types. Specifies the dimensionality of a texture.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum TextureType {
    /// 2D Texture.
    D2,
    /// Cubemap Texture.
    Cube,
    /// 3D Texture.
    D3,
}

use nvtt_sys::NvttTextureType;
impl From<TextureType> for NvttTextureType {
    fn from(val: TextureType) -> Self {
        match val {
            TextureType::D2 => NvttTextureType::NVTT_TextureType_2D,
            TextureType::Cube => NvttTextureType::NVTT_TextureType_Cube,
            TextureType::D3 => NvttTextureType::NVTT_TextureType_3D,
        }
    }
}

impl From<NvttTextureType> for TextureType {
    fn from(other: NvttTextureType) -> Self {
        match other {
            NvttTextureType::NVTT_TextureType_2D => Self::D2,
            NvttTextureType::NVTT_TextureType_Cube => Self::Cube,
            NvttTextureType::NVTT_TextureType_3D => Self::D3,
        }
    }
}

/// Specifies how to handle coordinates outside the typical image range. For use with
/// [`Surface::set_wrap_mode()`]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum WrapMode {
    /// Coordinates are clamped, moving them to the closest coordinate inside the image.
    Clamp,
    /// The image is treated as if it repeats on both axes, mod each dimension. For instance, for a 4x4 image, `(5, -2)` wraps to `(1, 2)`.
    Repeat,
    /// Coordinates are treated as if they reflect every time they pass through the center of an edge texel. For instance, for a 10x10 image, `(8, 0)`, `(10, 0)`, `(26, 0)`, and `(28, 0)` all mirror to `(8, 0)`.
    Mirror,
}

use nvtt_sys::NvttWrapMode;
impl From<WrapMode> for NvttWrapMode {
    fn from(val: WrapMode) -> Self {
        match val {
            WrapMode::Clamp => NvttWrapMode::NVTT_WrapMode_Clamp,
            WrapMode::Repeat => NvttWrapMode::NVTT_WrapMode_Repeat,
            WrapMode::Mirror => NvttWrapMode::NVTT_WrapMode_Mirror,
        }
    }
}

impl From<NvttWrapMode> for WrapMode {
    fn from(other: NvttWrapMode) -> Self {
        match other {
            NvttWrapMode::NVTT_WrapMode_Clamp => Self::Clamp,
            NvttWrapMode::NVTT_WrapMode_Repeat => Self::Repeat,
            NvttWrapMode::NVTT_WrapMode_Mirror => Self::Mirror,
        }
    }
}

/// A generic filter. Can be either a mipmap filter or a resize filter.
#[derive(Clone, Copy, Debug)]
pub struct Filter<T> {
    /// Filter width.
    pub width: f32,
    /// Filtering algorithm.
    pub algorithm: T,
}

impl Filter<Mipmap> {
    /// Returns the default [`Mipmap::Box`] filter, with `width = 0.5`.
    pub const fn mipmap_box() -> Self {
        Self {
            width: 0.5,
            algorithm: Mipmap::Box,
        }
    }

    /// Returns the default [`Mipmap::Triangle`] filter, with `width = 1.0`.
    pub const fn mipmap_triangle() -> Self {
        Self {
            width: 1.0,
            algorithm: Mipmap::Triangle,
        }
    }

    /// Returns the default [`Mipmap::Kaiser`] filter, with `width = 3.0`, `alpha = 4.0`, and `freq = 1.0`.
    pub const fn mipmap_kaiser() -> Self {
        Self {
            width: 3.0,
            algorithm: Mipmap::Kaiser {
                alpha: 4.0,
                freq: 1.0,
            },
        }
    }

    pub(crate) fn params(&self) -> [f32; 2] {
        let mut params = [0f32; 2];
        match self.algorithm {
            Mipmap::Box | Mipmap::Triangle => (),
            Mipmap::Kaiser { alpha, freq } => {
                params[0] = alpha;
                params[1] = freq;
            }
        }
        params
    }

    pub(crate) fn params_ptr(&self, params: &[f32; 2]) -> *const f32 {
        match self.algorithm {
            Mipmap::Box | Mipmap::Triangle => std::ptr::null(),
            Mipmap::Kaiser { .. } => params.as_ptr(),
        }
    }
}

impl Filter<Resize> {
    /// Returns the default [`Resize::Box`] filter, with `width = 0.5`.
    pub const fn box_resize() -> Self {
        Self {
            width: 0.5,
            algorithm: Resize::Box,
        }
    }

    /// Returns the default [`Resize::Triangle`] filter, with `width = 1.0`.
    pub const fn triangle_resize() -> Self {
        Self {
            width: 1.0,
            algorithm: Resize::Triangle,
        }
    }

    /// Returns the default [`Resize::Kaiser`] filter, with `width = 3.0`, `alpha = 4.0`, and `freq = 1.0`.
    pub const fn kaiser_resize() -> Self {
        Self {
            width: 3.0,
            algorithm: Resize::Kaiser {
                alpha: 4.0,
                freq: 1.0,
            },
        }
    }

    /// Returns the default [`Resize::Mitchell`] filter, with `width = 2.0`, `b = 1.0 / 3.0`, and `c = 2.0 / 3.0`.
    pub const fn mitchell_resize() -> Self {
        Self {
            width: 2.0,
            algorithm: Resize::Mitchell {
                b: 0.33333333,
                c: 0.666_666_7,
            },
        }
    }

    pub(crate) fn params(&self) -> [f32; 2] {
        let mut params = [0f32; 2];
        match self.algorithm {
            Resize::Box | Resize::Triangle => (),
            Resize::Kaiser { alpha, freq } => {
                params[0] = alpha;
                params[1] = freq;
            }
            Resize::Mitchell { b, c } => {
                params[0] = b;
                params[1] = c;
            }
        }
        params
    }

    pub(crate) fn params_ptr(&self, params: &[f32; 2]) -> *const f32 {
        match self.algorithm {
            Resize::Box | Resize::Triangle => std::ptr::null(),
            Resize::Kaiser { .. } => params.as_ptr(),
            Resize::Mitchell { .. } => params.as_ptr(),
        }
    }
}

/// A mipmap filter. For use with [`Surface::build_next_mipmap()`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mipmap {
    /// Box filter is quite good and very fast. It has some special paths for downsampling
    /// by exactly a factor of `2`. `filter_width` defaults to `0.5`; `box(x)` is equal to `1`
    /// when `|x| < filter_width` and `0` otherwise.
    Box,
    /// Triangle filter blurs the results too much, but that might be what you want.
    /// `filter_width` defaults to `1.0`; `triangle(x)` is equal to `filter_width - |x|`
    /// when `|x|< filter_width` and `0` otherwise.
    Triangle,
    /// Kaiser-windowed Sinc filter is the best downsampling filter, and close to a mathematically ideal windowing filter.
    /// If the window size is too large, it can introduce ringing.
    ///
    /// `filter_width` controls the width of the Kaiser window.
    /// Larger values take longer to compute and include more oscillations of the sinc filter.
    Kaiser {
        /// The sharpness of the Kaiser window. Higher values make the main lobe wider, but reduce sideband energy.
        alpha: f32,
        /// The frequency of the sinc filter. Higher values include higher frequencies.
        freq: f32,
    },
}

impl Mipmap {
    /// The default filter width for a given mipmap filter.
    pub fn filter_width_default(self) -> f32 {
        match self {
            Self::Box => 0.5,
            Self::Triangle => 1.0,
            Self::Kaiser { .. } => 3.0,
        }
    }

    /// The default [`Mipmap::Kaiser`] filter parameters.
    pub fn kaiser_default() -> Self {
        Self::Kaiser {
            alpha: 4.0,
            freq: 1.0,
        }
    }
}

use nvtt_sys::NvttMipmapFilter;
impl From<Mipmap> for NvttMipmapFilter {
    fn from(val: Mipmap) -> Self {
        match val {
            Mipmap::Box => NvttMipmapFilter::NVTT_MipmapFilter_Box,
            Mipmap::Triangle => NvttMipmapFilter::NVTT_MipmapFilter_Triangle,
            Mipmap::Kaiser { .. } => NvttMipmapFilter::NVTT_MipmapFilter_Kaiser,
        }
    }
}

/// A resize filter. For use with [`Surface::resize_filtered()`] and [`Surface::resize_make_square()`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Resize {
    /// See [`Mipmap::Box`].
    Box,
    /// See [`Mipmap::Triangle`].
    Triangle,
    /// See [`Mipmap::Kaiser`].
    Kaiser { alpha: f32, freq: f32 },
    /// Mitchell & Netravali's two parameter cubic filter.
    ///
    /// `filter_width` (default: `2.0`) can truncate the filter, but should usually be left at the default.
    Mitchell {
        /// Defaults to `1.0 / 3.0`.
        b: f32,
        /// Defaults to `2.0 / 3.0`.
        c: f32,
    },
}

use nvtt_sys::NvttResizeFilter;
impl From<Resize> for NvttResizeFilter {
    fn from(val: Resize) -> Self {
        match val {
            Resize::Box => NvttResizeFilter::NVTT_ResizeFilter_Box,
            Resize::Triangle => NvttResizeFilter::NVTT_ResizeFilter_Box,
            Resize::Kaiser { .. } => NvttResizeFilter::NVTT_ResizeFilter_Kaiser,
            Resize::Mitchell { .. } => NvttResizeFilter::NVTT_ResizeFilter_Mitchell,
        }
    }
}

/// Tone mapping functions. For use with [`Surface::tonemap()`]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ToneMapper {
    /// Colors inside `[0,1)^3` are preserved; colors outside are tone mapped using `(r', g', b') = (r, g, b)/max(r, g, b)`.
    /// This clamps colors to the RGB cube, but preserves hue. It is not invertible.
    Linear,
    /// Applies a Reinhard operator to each channel: `c' = c / (c + 1)`.
    Reinhard,
    /// Applies an exponential tone mapper to each channel: `c' = 1 - 2^(-c)`.
    Halo,
}

use nvtt_sys::NvttToneMapper;
impl From<ToneMapper> for NvttToneMapper {
    fn from(val: ToneMapper) -> Self {
        match val {
            ToneMapper::Linear => NvttToneMapper::NVTT_ToneMapper_Linear,
            ToneMapper::Reinhard => NvttToneMapper::NVTT_ToneMapper_Reinhard,
            ToneMapper::Halo => NvttToneMapper::NVTT_ToneMapper_Halo,
        }
    }
}

/// Specifies a normal transformation. For use with [`Surface::transform_normals()`] and [`Surface::reconstruct_normals()`].
///
/// Used to store 3D `(x, y, z)` normals in 2D `(x, y)`.
///
/// We define these in terms of their 2D -> 3D reconstructions, since their transformations
/// are the inverse of the reconstructions. Most require `z >= 0.0f`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum NormalTransform {
    /// Reconstructs the `z` component using `z = sqrt(1 - x^2 + y^2)`.
    Orthographic,
    /// Stereographic projection (like looking from the bottom of the sphere of normals and
    /// projecting points onto a plane at `z = 1`). Reconstructed using
    ///
    /// ```text
    /// d = 2/(1 + min(x^2 + y^2, 1));
    /// return (x*d, y*d, d-1);
    /// ```
    Stereographic,
    /// Reconstructed using `normalize(x, y, 1 - min(x^2 + y^2, 1))`.
    Paraboloid,
    /// Reconstructed using `normalize(x, y, (1-x^2)(1-y^2))`.
    Quartic,
}

use nvtt_sys::NvttNormalTransform;
impl From<NormalTransform> for NvttNormalTransform {
    fn from(val: NormalTransform) -> Self {
        match val {
            NormalTransform::Orthographic => NvttNormalTransform::NVTT_NormalTransform_Orthographic,
            NormalTransform::Stereographic => {
                NvttNormalTransform::NVTT_NormalTransform_Stereographic
            }
            NormalTransform::Paraboloid => NvttNormalTransform::NVTT_NormalTransform_Paraboloid,
            NormalTransform::Quartic => NvttNormalTransform::NVTT_NormalTransform_Quartic,
        }
    }
}

/// Swizzle order. For use with [`Surface::swizzle()`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Swizzle {
    /// Set to the current red channel.
    R = 0,
    /// Set to the current green channel.
    G = 1,
    /// Set to the current blue channel.
    B = 2,
    /// Set to the current alpha channel.
    A = 3,
    /// Set entire channel to `-1.0`.
    NegOne = 4,
    /// Set entire channel to `0.0`.
    Zero = 5,
    /// Set entire channel to `1.0`.
    One = 6,
}

/// Extents rounding mode. For use with [`Surface::resize_rounded()`] and
/// [`Surface::resize_make_square()`].
///
/// Determines how to round sizes to different sets when shrinking an image.
///
/// For each of the `PowerOfTwo` modes, `max_extent` is first rounded to the previous power of two.
///
/// Then all extents are scaled and truncated without changing the aspect ratio, using `s = max((s * maxExtent) / m, 1)`, where `m` is the maximum width, height, or depth.
///
/// If the texture is a cube map, the width and height are then averaged to make the resulting texture square.
///
/// Finally, extents are rounded to a set of possible sizes depending on this enum.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RoundMode {
    /// Each extent is left as-is.
    None,
    /// Each extent is rounded up to the next power of two.
    ToNextPowerOfTwo,
    /// Each extent is rounded either up or down to the nearest power of two.
    ToNearestPowerOfTwo,
    /// Each element is rounded down to the next power of two.
    ToPreviousPowerOfTwo,
}

use nvtt_sys::NvttRoundMode;
impl From<RoundMode> for NvttRoundMode {
    fn from(val: RoundMode) -> Self {
        match val {
            RoundMode::None => NvttRoundMode::NVTT_RoundMode_None,
            RoundMode::ToNextPowerOfTwo => NvttRoundMode::NVTT_RoundMode_ToNextPowerOfTwo,
            RoundMode::ToNearestPowerOfTwo => NvttRoundMode::NVTT_RoundMode_ToNearestPowerOfTwo,
            RoundMode::ToPreviousPowerOfTwo => NvttRoundMode::NVTT_RoundMode_ToPreviousPowerOfTwo,
        }
    }
}

/// Face of a cubemap. For use with [`CubeSurface::face()`]
#[repr(i32)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CubeFace {
    /// Positive X face, index 0.
    PosX = 0,
    /// Negative X face, index 1.
    NegX = 1,
    /// Positive Y face, index 2.
    PosY = 2,
    /// Negative Y face, index 3.
    NegY = 3,
    /// Positive Z face, index 4.
    PosZ = 4,
    /// Negative Z face, index 5.
    NegZ = 5,
}

/// Specifies how to fold or unfold a cube map from or to a 2D texture.
/// For use with [`CubeSurface::fold()`] and [`CubeSurface::unfold()`]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CubeLayout {
    /// A `3*edgeLength (width) x 4*edgeLength` texture, laid out as follows:
    ///
    /// |        | PosY/2 |        |
    /// |--------|--------|--------|
    /// | NegX/1 | PosZ/4 | PosX/0 |
    /// |        | NegY/3 |        |
    /// |        | NegZ/5 |        |
    ///
    /// Face 5 is rotated 180 degrees.
    VerticalCross,
    /// A `4*edgeLength (width) x 3*edgeLength` texture, laid out as follows:
    ///
    /// |        | PosY/2 |        |        |
    /// |--------|--------|--------|--------|
    /// | NegX/1 | PosZ/4 | PosX/0 | NegZ/5 |
    /// |        | NegY/3 |        |        |
    ///
    /// Face 5 is rotated 180 degrees.
    HorizontalCross,
    /// A column layout, laid out as follows:
    ///
    /// | PosX/0 |
    /// |--------|
    /// | NegX/1 |
    /// | PosY/2 |
    /// | NegY/3 |
    /// | PosZ/4 |
    /// | NegZ/5 |
    Column,
    /// A row layout, laid out as follows:
    ///
    /// | PosX/0 | NegX/1 | PosY/2 | NegY/3 | PosZ/4 | PosZ/5 |
    /// |--------|--------|--------|--------|--------|--------|
    Row,
}

impl CubeLayout {
    /// Returns true if the dimensions are supported by this layout. Always returns false if `w` or
    /// `h` are `0`.
    pub fn dim_supported(&self, w: u32, h: u32) -> bool {
        match self {
            Self::VerticalCross => {
                let w_supported = (w % 3 == 0) && w > 0;
                let h_supported = (h % 4 == 0) && h > 0;

                if w_supported && h_supported {
                    let w_subimage = w / 3;
                    let h_subimage = h / 4;

                    w_subimage == h_subimage
                } else {
                    false
                }
            }
            Self::HorizontalCross => {
                let w_supported = (w % 4 == 0) && w > 0;
                let h_supported = (h % 3 == 0) && h > 0;

                if w_supported && h_supported {
                    let w_subimage = w / 4;
                    let h_subimage = h / 3;

                    w_subimage == h_subimage
                } else {
                    false
                }
            }
            Self::Column => {
                let w_supported = w > 0;
                let h_supported = (h % 6 == 0) && h > 0;

                if w_supported && h_supported {
                    let w_subimage = w;
                    let h_subimage = h / 6;

                    w_subimage == h_subimage
                } else {
                    false
                }
            }
            Self::Row => {
                let w_supported = (w % 6 == 0) && w > 0;
                let h_supported = h > 0;

                if w_supported && h_supported {
                    let w_subimage = w / 6;
                    let h_subimage = h;

                    w_subimage == h_subimage
                } else {
                    false
                }
            }
        }
    }
}

impl From<CubeLayout> for nvtt_sys::NvttCubeLayout {
    fn from(val: CubeLayout) -> Self {
        match val {
            CubeLayout::VerticalCross => nvtt_sys::NvttCubeLayout::NVTT_CubeLayout_VerticalCross,
            CubeLayout::HorizontalCross => {
                nvtt_sys::NvttCubeLayout::NVTT_CubeLayout_HorizontalCross
            }
            CubeLayout::Column => nvtt_sys::NvttCubeLayout::NVTT_CubeLayout_Column,
            CubeLayout::Row => nvtt_sys::NvttCubeLayout::NVTT_CubeLayout_Row,
        }
    }
}
