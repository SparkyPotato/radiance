import graph;
import common;

[shader("miss")]
void main(inout ShadowPayload payload) {
	payload.shadowed = false;
}
