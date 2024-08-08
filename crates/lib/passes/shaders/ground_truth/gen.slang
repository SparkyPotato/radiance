import graph.types;
import graph.util.rng;
import common;

func gen_ray(inout rng: Rng) -> RayDesc {
	let pixel = DispatchRaysIndex().xy;
	let total = DispatchRaysDimensions().xy;
	let cam = *Constants.inv_camera;

	RayDesc ray;
	let pix = f32x2(pixel) + rng.sample2();
	let clip = pix / f32x2(total) * 2.f - 1.f;
	ray.Origin = mul(cam.view, float4(0.f, 0.f, 0.f, 1.f)).xyz;
	let target = normalize(mul(cam.proj, float4(clip.x, -clip.y, 1.f, 1.f)).xyz);
	ray.Direction = mul(cam.view, float4(target, 0.f)).xyz;
	ray.TMin = 0.001f;
	ray.TMax = 10000.f;

	return ray;
}

func write_samples(acc: f32x3, sample_count: u32) {
	let pixel = DispatchRaysIndex().xy;
	let value = Constants.img.load(pixel).xyz;
	let samples = Constants.samples;
	let p1 = samples + sample_count;
	Constants.img.store(pixel, f32x4((samples * value + acc) / p1, 1.f));
}

func lum(col: f32x3) -> f32 {
	let mul = col * f32x3(0.2126f, 0.7152f, 0.0722f);
	return mul.x + mul.y + mul.z;
}

static const u32 SAMPLE_COUNT = 10;

[shader("raygeneration")]
void main() {
	let pixel = DispatchRaysIndex().xy;
	let total = DispatchRaysDimensions().xy;	
	var rng = Constants.rng.init(pixel.y * total.x + pixel.x);
	
	f32x3 acc = 0.f;
	var samples = SAMPLE_COUNT;
	for (int i = 0; i < SAMPLE_COUNT; i++) {
		f32x3 b = 1.f;
		var ray = gen_ray(rng);
		var specular = true;
		f32x3 sample = 0.f;
		for (int bounces = 0; bounces < 16; bounces++) {
			var p = Payload(rng, specular);
			Constants.as.trace(RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, p);
			rng = p.rng;
			specular = p.specular;

			let val = b * p.radiance;
			if (any(or(isinf(val), isnan(val)))) {
				samples--;
				break;
			}
			sample += val;
			if (!p.hit || all(p.color == 0.f) || p.pdf == 0.f) break;

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

		acc += sample;
	}

	write_samples(acc, samples);
}

