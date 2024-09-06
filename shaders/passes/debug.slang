import graph;
import graph.util;
import passes.asset;
import passes.mesh.cull;
import passes.mesh.visbuffer;

public struct PushConstants {
	public Instance* instances;
	public Camera* camera;
	public MeshletQueue early_hw;
	public MeshletQueue early_sw;
	public MeshletQueue late_hw;
	public MeshletQueue late_sw;
	public VisBufferTex read;
	public u32 bottom;
	public u32 top;
};

[vk::push_constant]
public PushConstants Constants;

// Stolen from https://gist.github.com/badboy/6267743 and niagara.
// Thanks copilot.
public u32 hash(u32 a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

NodePointer get_meshlet(u32 id) {
	let early_hw = Constants.early_hw.len();
	let early_sw = Constants.early_sw.len();
	let late_hw = Constants.late_hw.len();
	let late_sw = Constants.late_sw.len();
	if (id < early_hw)
		return Constants.early_hw.get(id);
	id -= early_hw;
	if (id < early_sw)
		return Constants.early_sw.get(id);
	id -= early_sw;
	if (id < late_hw)
		return Constants.late_hw.get(id);
	id -= late_hw;
	return Constants.late_sw.get(id);
}

[shader("pixel")]
f32x4 triangles(ScreenOutput input) : SV_Target0 {
	let visbuf = Constants.read.decode(input.uv);
	if (visbuf.data == 0xffffffff)
		return f32x4(0.f, 0.f, 0.f, 1.f);
	let data = VisBufferData(visbuf.data);
	let h = hash(data.triangle_id);
	let color = f32x3(f32(h & 255), f32((h >> 8) & 255), f32((h >> 16) & 255));
	return f32x4(color / 255.f, 1.f);
}

[shader("pixel")]
f32x4 meshlets(ScreenOutput input) : SV_Target0 {
	let visbuf = Constants.read.decode(input.uv);
	if (visbuf.data == 0xffffffff)
		return f32x4(0.f, 0.f, 0.f, 1.f);
	let data = VisBufferData(visbuf.data);
	NodePointer p = get_meshlet(data.meshlet_id);
	let h = hash(p.instance) ^ hash(p.node_offset);
	let color = f32x3(f32(h & 255), f32((h >> 8) & 255), f32((h >> 16) & 255));
	return f32x4(color / 255.f, 1.f);
}

f32x3 heatmap(f32 heat) {
	f32x3 cols[] = {
		f32x3(0.f, 0.f, 0.f), f32x3(0.f, 0.f, 1.f), f32x3(0.f, 1.f, 1.f), f32x3(0.f, 1.f, 0.f),
		f32x3(1.f, 1.f, 0.f), f32x3(1.f, 0.f, 0.f), f32x3(1.f, 1.f, 1.f),
	};
	let scaled = heat * (cols.getCount() - 1.f);
	let bot = u32(floor(scaled));
	let top = u32(ceil(scaled));
	return lerp(cols[bot], cols[top], scaled - f32(bot));
}

[shader("pixel")]
f32x4 overdraw(ScreenOutput input) : SV_Target0 {
	let value = Constants.read.overdraw(input.uv);
	let heat = clamp((f32(value) - f32(Constants.bottom)) / f32(Constants.top - Constants.bottom), 0.f, 1.f);
	return f32x4(heatmap(heat), 1.f);
}

[shader("pixel")]
f32x4 hwsw(ScreenOutput input) : SV_Target0 {
	let data = Constants.read.hwsw(input.uv);

	f32x3 cols[] = { f32x3(0.f, 0.f, 0.f), f32x3(0.f, 1.f, 0.f), f32x3(1.f, 1.f, 0.f), f32x3(1.f, 0.f, 0.f) };
	if (data >= cols.getCount())
		return f32x4(1.f, 0.f, 1.f, 1.f);
	else
		return f32x4(cols[data], 1.f);
}

[shader("pixel")]
f32x4 normals(ScreenOutput input) : SV_Target0 {
	let visbuf = Constants.read.decode(input.uv);
	if (visbuf.data == 0xffffffff)
		return f32x4(0.f, 0.f, 0.f, 1.f);
	let data = VisBufferData(visbuf.data);
	NodePointer p = get_meshlet(data.meshlet_id);

	let instance = Constants.instances[p.instance];
	let meshlet = instance.meshlet(p.node_offset);
	let camera = Constants.camera[0];
	let mvp = mul(camera.view_proj, instance.get_transform());

	let t = meshlet.tri(instance.mesh, data.triangle_id);
	let v0f = meshlet.vertex(instance.mesh, t.x);
	let v1f = meshlet.vertex(instance.mesh, t.y);
	let v2f = meshlet.vertex(instance.mesh, t.z);
	let v0 = transform_vertex(mvp, v0f).uv;
	let v1 = transform_vertex(mvp, v1f).uv;
	let v2 = transform_vertex(mvp, v2f).uv;
	let depth = visbuf.depth;

	let a = v1.xy - v0.xy;
	let b = v2.xy - v0.xy;
	let c = input.uv - v0.xy;
	let d00 = dot(a, a);
	let d01 = dot(a, b);
	let d11 = dot(b, b);
	let d20 = dot(c, a);
	let d21 = dot(c, b);
	let denom = d00 * d11 - d01 * d01;
	let v = (d11 * d20 - d01 * d21) / denom;
	let w = (d00 * d21 - d01 * d20) / denom;
	let u = 1.f - v - w;

	let norm = depth * (v0f.normal * u / v0.z + v1f.normal * v / v1.z + v2f.normal * w / v2.z);
	return f32x4(abs(norm), 1.f);
}

[shader("pixel")]
f32x4 hzb_mip(ScreenOutput input) : SV_Target0 {
	let visbuf = Constants.read.decode(input.uv);
	if (visbuf.data == 0xffffffff)
		return f32x4(0.f, 0.f, 0.f, 1.f);
	let data = VisBufferData(visbuf.data);
	NodePointer p = get_meshlet(data.meshlet_id);

	let instance = Constants.instances[p.instance];
	let meshlet = instance.meshlet(p.node_offset);
	let camera = Constants.camera[0];
	let mvp = mul(camera.view_proj, instance.get_transform());

	if (let aabb = project_aabb(mvp, camera.near, meshlet.aabb)) {
		let size = Constants.read.size();
		let asize = (aabb.max - aabb.min).xy * size * 0.5f;
		let heat = log2(max(asize.x, asize.y)) / log2(max(size.x, size.y));
		return f32x4(heatmap(heat), 1.f);
	}
	return f32x4(1.f, 0.f, 1.f, 1.f);
}

[shader("pixel")]
f32x4 hzb_uv(ScreenOutput input) : SV_Target0 {
	let visbuf = Constants.read.decode(input.uv);
	if (visbuf.data == 0xffffffff)
		return f32x4(0.f, 0.f, 0.f, 1.f);
	let data = VisBufferData(visbuf.data);
	NodePointer p = get_meshlet(data.meshlet_id);

	let instance = Constants.instances[p.instance];
	let meshlet = instance.meshlet(p.node_offset);
	let camera = Constants.camera[0];
	let mvp = mul(camera.view_proj, instance.get_transform());

	if (let aabb = project_aabb(mvp, camera.near, meshlet.aabb)) {
		let us = f32x3(input.uv, visbuf.depth);
		let norm = (us - aabb.min) / (aabb.max - aabb.min);
		return f32x4(norm, 1.f);
	}
	return f32x4(0.f, 0.f, 0.f, 1.f);
}
