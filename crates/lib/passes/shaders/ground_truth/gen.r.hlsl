#include "common.l.hlsl"

[shader("raygeneration")]
void main() {
	uint2 pixel = DispatchRaysIndex().xy;
	float4 color = float4(1.f, 0.f, 0.f, 1.f);
	RWTexture2Ds[Constants.img][pixel] = color;
}

