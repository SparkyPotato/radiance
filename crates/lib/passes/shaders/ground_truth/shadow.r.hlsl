#include "common.l.hlsl"

[shader("miss")]
void main(inout ShadowPayload payload) {
	payload.unshadowed = true;
}

