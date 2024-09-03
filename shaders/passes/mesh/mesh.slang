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
	bool culled : SV_CullPrimitive;
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

struct Init {
	u32 mid;
	u8* mesh;
	Meshlet meshlet;
	f32x4x4 mvp;

	__init(u32 gid, constexpr bool hw) {
		NodePointer p;
		if (EARLY) {
			if (hw) {
				p = Constants.early_hw.get(gid);
				this.mid = gid;
			} else {
				p = Constants.early_sw.get(gid);
				this.mid = Constants.early_hw.len() + gid;
			}
		} else {
			if (hw) {
				p = Constants.late_hw.get(gid);
				this.mid = Constants.early_hw.len() + Constants.early_sw.len() + gid;
			} else {
				p = Constants.late_sw.get(gid);
				this.mid = Constants.early_hw.len() + Constants.early_sw.len() + Constants.late_hw.len() + gid;
			}
		}
		let instance = Constants.instances[p.instance];
		this.mesh = instance.mesh;
		this.meshlet = instance.meshlet(p.node_offset);
		this.mvp = mul(Constants.camera[0].view_proj, instance.get_transform());
	}

	VertexTransform transform(u32 gtid) {
		return transform_vertex(this.mvp, this.meshlet.vertex(this.mesh, gtid));
	}

	u32x3 tri(u32 gtid) {
		return this.meshlet.tri(this.mesh, gtid);
	}
}

f32 edge_fn(f32x2 a, f32x2 b, f32x2 c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

groupshared f32x3 Pos[128];

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
		Pos[gtid] = v.uv;
	}
	workgroup_sync_barrier();

	if (gtid < tri_count) {
		let i = init.tri(gtid);
		triangles[gtid] = i;
		VisBufferData data = { init.mid, gtid };
		// TODO: not strictly correct
		let backface = edge_fn(Pos[i.x].xy, Pos[i.y].xy, Pos[i.z].xy) < 0.f;
		visbuffer[gtid] = { data.encode(), backface };
	}
}

f32x3 uv_vert(Init init, u32 gtid) {
	let v = init.transform(gtid);
	let dim = f32x2(Constants.output.size());
	return f32x3(v.uv.xy * dim, v.uv.z);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sw(u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex) {
	let init = Init(gid, false);
	let vert_count = init.meshlet.vertex_count;
	let tri_count = init.meshlet.tri_count;
	let dim = Constants.output.size();

	if (gtid < vert_count) {
		Pos[gtid] = uv_vert(init, gtid);
	}
	GroupMemoryBarrierWithGroupSync();
	return;

	if (gtid >= tri_count)
		return;

	let t = init.tri(gtid);
	let v0 = Pos[t.x];
	let v1 = Pos[t.y];
	let v2 = Pos[t.z];
	VisBufferData data = { init.mid, gtid };
	let write = data.encode();

	let mi = min3(v0, v1, v2).xy;
	let ma = max3(v0, v1, v2).xy;
	var minv = i32x2(floor(mi));
	var maxv = i32x2(floor(ma));
	minv = max(minv, i32x2(0, 0));
	maxv = min(maxv, i32x2(dim - 1));
	maxv = min(maxv, minv + 127);  // Try not to TDR
	if (any(minv > maxv))
		return;

	let w_x = f32x3(v1.y - v2.y, v2.y - v0.y, v0.y - v1.y);
	let w_y = f32x3(v2.x - v1.x, v0.x - v2.x, v1.x - v0.x);
	let par_area = edge_fn(v0.xy, v1.xy, v2.xy);
	if (par_area < 0.f)
		return;
	let v_z = f32x3(v0.z, v1.z, v2.z) / par_area;
	let z_x = dot(v_z, w_x);
	let z_y = dot(v_z, w_y);

	let start = minv + 0.5f;
	var w_row = f32x3(edge_fn(v1.xy, v2.xy, start), edge_fn(v2.xy, v0.xy, start), edge_fn(v0.xy, v1.xy, start));
	var z_row = dot(v_z, w_row);
	if (WaveActiveAnyTrue(maxv.x - minv.x > 4)) {
		let e012 = -w_x;
		let oe = e012 < 0.f;
		let ie012 = select(e012 != 0.f, 1.f / e012, 1e8);
		let width = maxv.x - minv.x;
		for (int y = minv.y; y <= maxv.y; y++) {
			let cross_x = w_row * ie012;
			let min_x2 = select(oe, cross_x, 0.f);
			let max_x2 = select(oe, width, cross_x);
			var x0 = u32(ceil(max3(min_x2.x, min_x2.y, min_x2.z)));
			var x1 = u32(min3(max_x2.x, max_x2.y, max_x2.z));
			var w = w_row + w_x * f32(x0);
			var z = z_row + z_x * f32(x0);
			x0 += minv.x;
			x1 += minv.x;
			for (int x = x0; x <= x1; x++) {
				if (min3(w.x, w.y, w.z) >= 0.f)
					Constants.output.write(u32x2(x, y), z, write, true);
				w += w_x;
				z += z_x;
			}
			w_row += w_y;
			z_row += z_y;
		}
	} else {
		for (int y = minv.y; y <= maxv.y; y++) {
			var w = w_row;
			var z = z_row;
			for (int x = minv.x; x <= maxv.x; x++) {
				if (min3(w.x, w.y, w.z) >= 0.f)
					Constants.output.write(u32x2(x, y), z, write, true);
				w += w_x;
				z += z_x;
			}
			w_row += w_y;
			z_row += z_y;
		}
	}
}
