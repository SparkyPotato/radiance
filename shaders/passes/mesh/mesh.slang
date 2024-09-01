module mesh;

import graph;
import passes.asset;
import cull;
import visbuffer;

struct VertexOutput {
	f32x4 position : SV_Position;
};

struct PrimitiveOutput {
	[vk::location(0)]
	u32 data : VisBuffer;
}

struct PushConstants {
	Instance* instances;
	Camera* camera;
	MeshletQueue early_hw;
	MeshletQueue early_sw;
	MeshletQueue late_hw;
	MeshletQueue late_sw;
	VisBufferTex output;
}

[vk::push_constant]
PushConstants Constants;

NodePointer get(u32 gid, constexpr bool hw) {
	if (EARLY) {
		if (hw)
			return Constants.early_hw.get(gid);
		else
			return Constants.early_sw.get(gid);
	} else {
		if (hw)
			return Constants.late_hw.get(gid);
		else
			return Constants.late_sw.get(gid);
	}
}

u32 meshlet_id(u32 gid, constexpr bool hw) {
	if (EARLY) {
		if (hw)
			return gid;
		else
			return Constants.early_hw.len() + gid;
	} else {
		if (hw)
			return Constants.early_hw.len() + Constants.early_sw.len() + gid;
		else
			return Constants.early_hw.len() + Constants.early_sw.len() + Constants.late_hw.len() + gid;
	}
}

struct Init {
	u8* mesh;
	Meshlet meshlet;
	f32x4x4 mvp;
	u32 mid;

	__init(u32 gid, constexpr bool hw) {
		let p = get(gid, hw);
		let instance = Constants.instances[p.instance];
		this.mesh = instance.mesh;
		this.meshlet = instance.meshlet(p.node_offset);
		let camera = Constants.camera[0];
		this.mvp = mul(camera.view_proj, instance.get_transform());
		this.mid = meshlet_id(gid, hw);
	}

	VertexTransform transform(u32 gtid) {
		let v = this.meshlet.vertex(this.mesh, gtid);
		return transform_vertex(this.mvp, v);
	}

	u32x3 tri(u32 gtid) {
		return this.meshlet.tri(this.mesh, gtid);
	}
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void hw(u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex, out vertices VertexOutput vertices[128],
		out indices u32x3 triangles[124], out primitives PrimitiveOutput visbuffer[124]) {
	let init = Init(gid, true);
	let vert_count = init.meshlet.vertex_count;
	let tri_count = init.meshlet.tri_count;
	SetMeshOutputCounts(vert_count, tri_count);

	if (gtid < vert_count) {
		let v = init.transform(gtid);
		vertices[gtid].position = v.clip;
	}

	if (gtid < tri_count) {
		let i = init.tri(gtid);
		triangles[gtid] = i;
		VisBufferData data = { init.mid, gtid };
		visbuffer[gtid].data = data.encode();
	}
}

f32 edge_fn(f32x2 a, f32x2 b, f32x2 c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

f32x3 uv_vert(Init init, u32x2 dim, u32 gtid) {
	let v = init.transform(gtid);
	return f32x3(v.uv.xy * f32x2(dim), v.uv.z);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sw(u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex) {
	let init = Init(gid, false);
	let vert_count = init.meshlet.vertex_count;
	let tri_count = init.meshlet.tri_count;
	let dim = Constants.output.size();

	if (gtid >= tri_count)
		return;

	let t = init.tri(gtid);
	/// The vertex transformation should be done once and then stored in groupshared, but my driver crashes due to
	/// misalignment if i do that?
	let v0 = uv_vert(init, dim, t.x);
	let v1 = uv_vert(init, dim, t.y);
	let v2 = uv_vert(init, dim, t.z);
	VisBufferData data = { init.mid, gtid };
	let write = data.encode();

	let minv = max(min3(v0, v1, v2).xy, u32x2(0, 0));
	let maxv = min(max3(v0, v1, v2).xy, dim - 1);
	if (any(minv > maxv))
		return;

	let w_x = f32x3(v1.y - v2.y, v2.y - v0.y, v0.y - v1.y);
	let w_y = f32x3(v2.x - v1.x, v0.x - v2.x, v1.x - v0.x);
	let par_area = edge_fn(v0.xy, v1.xy, v2.xy);
	let vz = f32x3(v0.z, v1.z, v2.z) / par_area;
	let z_x = dot(vz, w_x);
	let z_y = dot(vz, w_y);

	let start = minv + 0.5f;
	var w_row = f32x3(edge_fn(v1.xy, v2.xy, start), edge_fn(v2.xy, v0.xy, start), edge_fn(v0.xy, v1.xy, start));
	var z_row = dot(vz, w_row);
	for (int y = minv.y; y <= maxv.y; y++) {
		var w = w_row;
		var z = z_row;
		for (int x = minv.x; x <= maxv.x; x++) {
			if (min3(w.x, w.y, w.z) >= 0.f)
				Constants.output.write(u32x2(x, y), min3(v0, v1, v2).z, write, true);

			w += w_x;
			z += z_x;
		}
		w_row += w_y;
		z_row += z_y;
	}
}
