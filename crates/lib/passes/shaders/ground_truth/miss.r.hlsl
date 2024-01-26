#include "common.l.hlsl"

[shader("miss")]
void main(inout Payload payload) {
	payload.value = float4(0.f, 0.f, 0.f, 1.f);
}

