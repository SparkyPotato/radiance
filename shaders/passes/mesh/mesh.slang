module mesh;

import graph;
import asset;
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
		this.mvp = mul(Constants.camera[0].view_proj, instance.transform.mat());
	}

	VertexTransform transform(u32 gtid) {
		return VertexTransform(this.mvp, this.meshlet.vertex(this.mesh, gtid));
	}

	u32x3 tri(u32 gtid) {
		return this.meshlet.tri(this.mesh, gtid);
	}
}

f32 edge_fn(f32x2 a, f32x2 b, f32x2 c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

groupshared f32x2 HWPos[128];

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
		HWPos[gtid] = v.uv.xy;
	}
	workgroup_sync_barrier();

	if (gtid < tri_count) {
		let i = init.tri(gtid);
		triangles[gtid] = i;
		VisBufferData data = { init.mid, gtid };
		// TODO: not strictly correct
		let backface = edge_fn(HWPos[i.x], HWPos[i.y], HWPos[i.z]) < 0.f;
		visbuffer[gtid] = { data.encode(), false };
	}
}

f32x3 uv_vert(Init init, u32 gtid) {
	let v = init.transform(gtid).uv;
	let dim = f32x2(Constants.output.size());
	return f32x3(v.xy * dim, v.z);
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
// https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
[shader("compute")]
[numthreads(128, 1, 1)]
void sw(u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex) {
	let init = Init(gid, false);
	let vert_count = init.meshlet.vertex_count;
	let tri_count = init.meshlet.tri_count;
	let dim = Constants.output.size();

	// for some reason, using groupshared memory to transform vertices causes a
	// GPU crash due to a misaligned memory access.
	// so i do this subgroup funkiness.
	u32x3 t;
	if (gtid < tri_count)
		t = init.tri(gtid);
	else
		t = 0;
	let verts = load_verts(init, t);

	if (gtid >= tri_count)
		return;

	let v0 = verts[0];
	let v1 = verts[1];
	let v2 = verts[2];
	VisBufferData data = { init.mid, gtid };
	let write = data.encode();

	let mi = min3(v0, v1, v2);
	let ma = max3(v0, v1, v2);
	var minv = i32x2(floor(mi.xy));
	var maxv = i32x2(floor(ma.xy));
	minv = max(minv, i32x2(0, 0));
	maxv = min(maxv, i32x2(dim - 1));
	maxv = min(maxv, minv + 31);  // Try not to TDR
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
	// saturates for top left rule.
	var w_row = f32x3(edge_fn(v1.xy, v2.xy, start) /* - saturate(w_x.x + saturate(1.f + w_y.x)) */,
					  edge_fn(v2.xy, v0.xy, start) /* - saturate(w_x.y + saturate(1.f + w_y.y)) */,
					  edge_fn(v0.xy, v1.xy, start) /* - saturate(w_x.z + saturate(1.f + w_y.z)) */);
	var z_row = dot(v_z, w_row);
	if (WaveActiveAnyTrue(maxv.x - minv.x > 4)) {
		scanline(minv, maxv, w_x, w_y, w_row, z_x, z_y, z_row, write);
	} else {
		bounding(minv, maxv, w_x, w_y, w_row, z_x, z_y, z_row, write);
	}
}

f32x3x3 load_verts(Init init, u32x3 t) {
	return f32x3x3(uv_vert(init, t.x), uv_vert(init, t.y), uv_vert(init, t.z));

	f32x3 verts[3];
	let min_vert = WaveActiveMin(min3(t.x, t.y, t.z));
	let max_vert = WaveActiveMax(max3(t.x, t.y, t.z));
	u32 wave_count = WaveGetLaneCount();
	[unroll]
	for (int base = min_vert; base <= max_vert; base += wave_count) {
		var i = base + WaveGetLaneIndex();
		i = select(i <= max_vert, i, 0);
		let v = uv_vert(init, i);

		[unroll]
		for (int j = 0; j < 3; j++) {
			let wave_index = i32(t[j]) - base;
			let valid = wave_index > 0 && wave_index < wave_count;
			let got = WaveReadLaneAt(v, select(valid, wave_index, 0));
			if (valid)
				verts[j] = got;
		}
	}
	return f32x3x3(verts[0], verts[1], verts[2]);
}

void scanline(i32x2 minv, i32x2 maxv, f32x3 w_x, f32x3 w_y, f32x3 w_row, f32 z_x, f32 z_y, f32 z_row, u32 write) {
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
				Constants.output.write(u32x2(x, y), z, write, 1);
			w += w_x;
			z += z_x;
		}
		w_row += w_y;
		z_row += z_y;
	}
}

void bounding(i32x2 minv, i32x2 maxv, f32x3 w_x, f32x3 w_y, f32x3 w_row, f32 z_x, f32 z_y, f32 z_row, u32 write) {
	for (int y = minv.y; y <= maxv.y; y++) {
		var w = w_row;
		var z = z_row;
		for (int x = minv.x; x <= maxv.x; x++) {
			if (min3(w.x, w.y, w.z) >= 0.f)
				Constants.output.write(u32x2(x, y), z, write, 2);
			w += w_x;
			z += z_x;
		}
		w_row += w_y;
		z_row += z_y;
	}
}
