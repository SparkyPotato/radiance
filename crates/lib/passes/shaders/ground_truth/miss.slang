import graph.types;
import common;

[shader("miss")]
func main(inout p: Payload) {
	p.hit = false;
	p.radiance = 0.5f * f32(p.specular);
}

