#include "radiance-asset-runtime/data.l.hlsl"
#include "radiance-passes/brdf/brdf.l.hlsl"

struct CameraData {
	float4x4 view;
	float4x4 proj;
};

struct PushConstants {
	u32 img;
	u32 samples;
	Buf<CameraData> inv_camera;
	Buf<Instance> instances;
	Buf<Material> materials;
	AS as;
	Sampler sampler;
	Rng rng;
};

PUSH PushConstants Constants;

struct [raypayload] Payload {
	float3 radiance     : read(caller) : write(miss, closesthit);
	bool hit            : read(caller) : write(miss, closesthit);
	float3 color        : read(caller) : write(closesthit);
	f32 pdf             : read(caller) : write(closesthit);
	float3 dir          : read(caller) : write(closesthit);
	float3 origin       : read(caller) : write(closesthit);
	float3 dot          : read(caller) : write(closesthit);
	bool specular : read(caller, miss, closesthit) : write(caller, closesthit);
	Rng rng : read(caller, closesthit) : write(caller, closesthit);

	static Payload init(Rng rng, bool specular) {
		Payload p;
		p.rng = rng;
		p.specular = specular;
		return p;
	}
};

struct [raypayload] ShadowPayload {
	bool unshadowed : read(caller) : write(caller, miss);
};

