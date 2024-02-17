#include "common.l.hlsl"

[shader("miss")]
void main(inout Payload p) {
	p.hit = false;
	p.radiance = 0.5f * p.specular;
}

