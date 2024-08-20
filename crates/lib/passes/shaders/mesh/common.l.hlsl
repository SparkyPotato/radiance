#pragma once

#include "radiance-passes/asset/data.l.hlsl"

struct VertexOutput {
	float4 position: SV_Position;
};

struct PrimitiveOutput {
	[[vk::location(0)]] u32 data: VisBuffer;
};
