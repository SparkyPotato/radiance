import graph.types;
import graph.as;
import graph.texture;
import graph.sampler;
import graph.util.rng;
import asset;

public struct CameraData {
	public float4x4 view;
	public float4x4 proj;
}

public struct PushConstants {
	public STex<f32x4, D2> img;
	public u32 samples;
	public CameraData* inv_camera;
	public Instance* instances;
	public Material* materials;
	public AS as;
	public Sampler sampler;
	public Rng rng;
}

public [vk::push_constant] PushConstants Constants;

public struct Payload {
	public f32x3 radiance;
	public bool hit;
	public f32x3 color;
	public f32 pdf;
	public f32x3 dir;
	public f32x3 origin;
	public f32x3 dot;
	public bool specular;
	public Rng rng;

	public __init(Rng rng, bool specular) {
		Payload p = {0};
		p.radiance = 0.f;
		p.hit = false;
		p.color = 0.f;
		p.pdf = 0.f;
		p.dir = 0.f;
		p.origin = 0.f;
		p.dot = 0.f;
		p.specular = specular;
		p.rng = rng;
		return p;
	}
};

public struct ShadowPayload {
	public bool unshadowed;
};
