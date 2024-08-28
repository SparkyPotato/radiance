#include "common.l.hlsl"

float4 main(VertexOutput input) : SV_Target0 {
	VisBuffer dec = Constants.read.decode(input.uv);
	VisBufferData data = VisBufferData::decode(dec.data);
	if (data.meshlet_id == 0xffffffff >> 7) discard;
	u32 id = data.meshlet_id;
	u32 len = Constants.early.len();
	MeshletPointer p;
	if (id < len) {
		p = Constants.early.get(id);
	} else {
		p = Constants.late.get(id - len);
	}

	Instance instance = Constants.instances.load(p.instance);
	Meshlet meshlet = instance.mesh.load<Meshlet>(p.meshlet_offset, 0);
	Camera camera = Constants.camera.load(0);
	float4x4 mvp = mul(camera.view_proj, instance.get_transform());

	u32 i = meshlet.tri(instance.mesh, data.triangle_id);
	uint3 t = uint3(i >> 0, i >> 8, i >> 16) & 0xff;
	Vertex v0f = meshlet.vertex(instance.mesh, t.x);
	Vertex v1f = meshlet.vertex(instance.mesh, t.y);
	Vertex v2f = meshlet.vertex(instance.mesh, t.z);
	float3 v0 = transform_vertex(mvp, v0f).uv;
	float3 v1 = transform_vertex(mvp, v1f).uv;
	float3 v2 = transform_vertex(mvp, v2f).uv;

	float2 a = v1.xy - v0.xy;
	float2 b = v2.xy - v0.xy;
	float2 c = input.uv - v0.xy;
	f32 d00 = dot(a, a);
	f32 d01 = dot(a, b);
	f32 d11 = dot(b, b);
	f32 d20 = dot(c, a);
	f32 d21 = dot(c, b);
	f32 denom = d00 * d11 - d01 * d01;
	f32 v = (d11 * d20 - d01 * d21) / denom;
	f32 w = (d00 * d21 - d01 * d20) / denom;
	f32 u = 1.f - v - w;

	float3 norm = dec.depth * (v0f.normal * u / v0.z + v1f.normal * v / v1.z + v2f.normal * w / v2.z);
	return float4(abs(norm), 1.f);
}