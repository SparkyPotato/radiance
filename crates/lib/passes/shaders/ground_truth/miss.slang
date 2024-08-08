import common;

[shader("miss")]
func main(inout p: Payload) {
	p.hit = false;
	p.radiance = 0.5f * p.specular;
}

