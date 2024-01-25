#include "radiance-core/interface.l.hlsl"

struct PushConstants {
	u32 img;
	Buf<float4x4> inv_camera;
	u32 as;
};

PUSH PushConstants Constants;

struct [raypayload] Payload {};

struct Attrs {};

