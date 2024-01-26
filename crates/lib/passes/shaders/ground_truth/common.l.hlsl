#include "radiance-passes/mesh/data.l.hlsl"

struct CameraData {
	float4x4 view;
	float4x4 proj;
};

struct PushConstants {
	u32 img;
	Buf<CameraData> inv_camera;
    Buf<Instance> instances;
	AS as;
};

PUSH PushConstants Constants;

struct [raypayload] Payload {
	float4 value : read(caller) : write(miss, closesthit);
};

struct [raypayload] ShadowPayload {
	bool unshadowed : read(caller) : write(caller, miss);
};

