import common;

[shader("miss")]
func main(inout ShadowPayload payload) {
	payload.unshadowed = true;
}

