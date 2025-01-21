import graph;
import common;

[shader("miss")]
void main(inout HitPayload p) {
	p.hit = false;
}
