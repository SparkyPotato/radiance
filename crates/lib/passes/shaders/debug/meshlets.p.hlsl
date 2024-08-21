#include "common.l.hlsl"

float4 main(VertexOutput input): SV_Target0 {
	uint2 pixel = Constants.visbuffer.pixel_of_uv(input.uv);
	u32 value = asuint(Constants.visbuffer.load(pixel).x);
	if (value == 0xffffffff) discard;

	VisBufferData data = VisBufferData::decode(value);
	u32 id = data.meshlet_id;
	u32 len = Constants.early.len();
	MeshletPointer p;
	if (id < len) {
		p = Constants.early.get(id);
	} else {
		p = Constants.late.get(id - len);
	}
	u32 h = hash(p.instance) ^ hash(p.meshlet_offset);
	float3 color = float3(float(h & 255), float((h >> 8) & 255), float((h >> 16) & 255));
	return float4(color / 255.0, 1.0);
}
