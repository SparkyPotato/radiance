# nvtt

[![crates.io](https://img.shields.io/crates/v/nvtt_rs.svg)](https://crates.io/crates/nvtt_rs)
[![docs](https://img.shields.io/badge/docs-orange.svg?style=flat)](https://mijalk0.github.io/nvtt)
[![rustc](https://img.shields.io/badge/rust-1.66%2B-orange.svg)](https://img.shields.io/badge/rust-1.66%2B-orange.svg)

A rust wrapper around the [Nvidia Texture Tools 3 library](https://developer.nvidia.com/gpu-accelerated-texture-compression).

NVTT 3 is a library that can be used to compress image data and files into compressed texture formats, and to handle compressed and uncompressed images.

In NVTT 3, most compression algorithms and image processing algorithms can be accelerated by the GPU. These have CPU fallbacks for GPUs without support for CUDA. CUDA operations can be enabled through the `cuda` feature.

# Dependencies

The NVTT 3 SDK must be installed on the system. A non-standard path to the binaries can be specified via the `NVTT_PATH` environment variable. A compiler supporting at least C99 and dynamic linking is also required.

## Windows

Windows 10 or 11 (64-bit) are required. The `Path` environment variable must also contain the path to the directory containing `nvtt.dll`. Note that this must be done manually, it is not done in a standard Nvidia Texture Tools install.

## Linux

64-bit only; Ubuntu 16.04+ or a similarly compatible distro is required. `libc.so` version 6 or higher is required as well.

# Limitations

Currently there is no file I/O support, no low-level (`nvtt_lowlevel.h`) wrapper, and no batch compression.

# Using nvtt

``` rust
// Create a surface
let input = InputFormat::Bgra8Ub {
    data: &[0u8; 16 * 16 * 4],
    unsigned_to_signed:  false,
};
let image = Surface::image(input, 16, 16, 1).unwrap();

// Create the compression context; enable CUDA if possible
let mut context = Context::new();
#[cfg(feature = "cuda")]
if *CUDA_SUPPORTED {
    context.set_cuda_acceleration(true);
}

// Specify compression settings to use; compress to Bc7
let mut compression_options = CompressionOptions::new();
compression_options.set_format(Format::Bc7);

// Specify how to write the compressed data; indicate as sRGB
let mut output_options = OutputOptions::new();
output_options.set_srgb_flag(true);

// Write the DDS header.
let header = context.output_header(
    &image,
    1, // number of mipmaps
    &compression_options,
    &output_options,
).unwrap();

// Compress and write the compressed data.
let bytes = context.compress(
    &image,
    &compression_options,
    &output_options,
).unwrap();

// Bc7 is 1 byte per pixel.
assert_eq!(16 * 16, bytes.len());
```

# License

Licensed under the MIT license. Note that the Nvidia Texture Tools SDK has its own separate license.
