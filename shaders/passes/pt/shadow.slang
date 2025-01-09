import graph;
import common;

[shader("miss")]
void main(inout ShadowPayload payload) {
	payload.unshadowed = true;
}
