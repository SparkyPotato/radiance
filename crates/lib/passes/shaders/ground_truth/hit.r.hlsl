#include "common.l.hlsl"

[shader("closesthit")]
void main(inout Payload payload, BuiltInTriangleIntersectionAttributes attr) {
	payload.value = 1.f;
}

