#include "common.l.hlsl"

RayDesc gen_ray(inout Rng rng) {
	uint2 pixel = DispatchRaysIndex().xy;
	uint2 total = DispatchRaysDimensions().xy;
	CameraData cam = Constants.inv_camera.load(0);

	RayDesc ray;
	float2 pix = float2(pixel) + rng.sample2();
	float2 clip = pix / float2(total) * 2.f - 1.f;
	ray.Origin = mul(cam.view, float4(0.f, 0.f, 0.f, 1.f)).xyz;
	float3 target = normalize(mul(cam.proj, float4(clip.x, -clip.y, 1.f, 1.f)).xyz);
	ray.Direction = mul(cam.view, float4(target, 0.f)).xyz;
	ray.TMin = 0.001f;
	ray.TMax = 10000.f;

	return ray;
}

static const u32 SAMPLE_COUNT = 2;

void write_samples(float3 acc) {
	uint2 pixel = DispatchRaysIndex().xy;
	float3 value = RWTexture2Ds[Constants.img][pixel].xyz;
	f32 samples = Constants.samples;
	f32 p1 = samples + SAMPLE_COUNT;
	RWTexture2Ds[Constants.img][pixel] = float4((samples * value + acc) / p1, 1.f);
}

f32 lum(float3 col) {
	float3 mul = col * float3(0.2126f, 0.7152f, 0.0722f);
	return mul.x + mul.y + mul.z;
}

[shader("raygeneration")]
void main() {
	uint2 pixel = DispatchRaysIndex().xy;
	uint2 total = DispatchRaysDimensions().xy;	
	Rng rng = Constants.rng.init(pixel.y * total.x + pixel.x);
	
	float3 acc = 0.f;
	for (int i = 0; i < SAMPLE_COUNT; i++) {
		float3 b = 1.f;
		RayDesc ray = gen_ray(rng);
		bool specular = true;
		for (u16 bounces = 0; bounces < 16; bounces++) {
			Payload p = Payload::init(rng, specular);
			TraceRay(ASes[Constants.as.index], RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, p);
			rng = p.rng;
			specular = p.specular;

			float3 val = b * p.radiance;
			if (any(or(isinf(val) ,isnan(val)))) break;
			acc += val;
			if (!p.hit || any(float4(p.color, p.pdf) == 0.f)) break;

			b *= p.color * p.dot / p.pdf;
			ray.Origin = p.origin;
			ray.Direction = p.dir;

			// рулетка
			if (bounces > 3) {
				f32 q = max(0.05f, 1.f - lum(b));
				if (rng.sample() < q) break;
				b /= 1.f - q;
			}
		}
	}

	write_samples(acc);
}

