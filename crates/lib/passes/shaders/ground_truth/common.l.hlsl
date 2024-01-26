#include "radiance-core/interface.l.hlsl"

struct Camera {
	float4x4 view;
	float4x4 proj;
};

struct PushConstants {
	u32 img;
	Buf<Camera> inv_camera;
	AS as;
};

PUSH PushConstants Constants;

struct [raypayload] Payload {
	float4 value : read(caller) : write(miss, closesthit);
};

