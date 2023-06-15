#pragma once

[[vk::binding(0, 0)]] RWByteAddressBuffer Buffers[];

[[vk::binding(1, 0)]] Texture1D Texture1Ds[];
[[vk::binding(1, 0)]] Texture2D Texture2Ds[];
[[vk::binding(1, 0)]] Texture3D Texture3Ds[];
[[vk::binding(1, 0)]] TextureCube TextureCubes[];

// [[vk::binding(2, 0)]] RWTexture2D RWTexture1Ds[];
// [[vk::binding(2, 0)]] RWTexture2D RWTexture2Ds[];
// [[vk::binding(2, 0)]] RWTexture2D RWTexture3Ds[];

[[vk::binding(3, 0)]] SamplerState Samplers[];
