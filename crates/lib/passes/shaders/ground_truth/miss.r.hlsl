#include "common.l.hlsl"

[shader("miss")]
void main(inout Payload payload) {
	payload.value = 1.f;
}

