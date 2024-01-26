#include "common.l.hlsl"

[shader("raygeneration")]
void main() {
	uint2 pixel = DispatchRaysIndex().xy;
	uint2 total = DispatchRaysDimensions().xy;

	CameraData cam = Constants.inv_camera.load(0);

	RayDesc ray;
	float2 clip = (float2(pixel) + 0.5f) / float2(total) * 2.f - 1.f;
	ray.Origin = mul(cam.view, float4(0.f, 0.f, 0.f, 1.f)).xyz;
	float3 target = normalize(mul(cam.proj, float4(clip.x, -clip.y, 1.f, 1.f)).xyz);
	ray.Direction = mul(cam.view, float4(target, 0.f)).xyz;
	ray.TMin = 0.001f;
	ray.TMax = 10000.f;

	Payload p;
	TraceRay(ASes[Constants.as.index], RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, p);

	RWTexture2Ds[Constants.img][pixel] = p.value;
}

