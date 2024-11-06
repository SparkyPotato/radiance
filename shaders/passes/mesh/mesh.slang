module mesh;

import graph;
import asset;
import cull;
import passes.visbuffer;

struct VertexOutput {
	f32x4 position : SV_Position;
};

struct PrimitiveOutput {
	[vk::location(0)]
	u32 data : VisBuffer;
	bool culled : SV_CullPrimitive;
}

public struct PushConstants {
	Instance* instances;
	Camera* camera;
	MeshletQueue queue;
	CullStats* stats;
	public VisBufferTex output;
}

[vk::push_constant]
public PushConstants Constants;

struct Init {
	u32 mid;
	u8* mesh;
	Meshlet meshlet;
	f32x4x4 mvp;

	__init(u32 gid, u32 gtid, constexpr bool hw) {
		NodePointer p;
		let report_stats = gid == 0 && gtid == 0;
		if (hw) {
			p = Constants.queue.get_hw(gid);
			this.mid = Constants.queue.hw_id(gid);
			if (report_stats)
				get_stats(Constants.stats)->hw_meshlets = Constants.queue.hw_count();
		} else {
			p = Constants.queue.get_sw(gid);
			this.mid = Constants.queue.sw_id(gid);
			if (report_stats)
				get_stats(Constants.stats)->sw_meshlets = Constants.queue.sw_count();
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

groupshared f32x4 Pos[128];

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void hw(u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex, out vertices VertexOutput vertices[128],
		out indices u32x3 triangles[124], out primitives PrimitiveOutput visbuffer[124]) {
	let init = Init(gid, gtid, true);
	let vert_count = init.meshlet.vertex_count;
	let tri_count = init.meshlet.tri_count;
	SetMeshOutputCounts(vert_count, tri_count);

	let dim = f32x2(Constants.output.size());
	if (gtid < vert_count) {
		let v = init.transform(gtid);
		vertices[gtid].position = v.clip;
		Pos[gtid] = f32x4(v.uv.xy * dim, v.uv.z, 1.f);
	}
	workgroup_sync_barrier();

	if (gtid < tri_count) {
		let tri = init.tri(gtid);
		triangles[gtid] = tri;
		VisBufferData data = { init.mid, gtid };
		visbuffer[gtid].data = data.encode();

		let a = Pos[tri.x].xyz;
		let b = Pos[tri.y].xyz;
		let c = Pos[tri.z].xyz;
		
		var culled = false;
		culled = culled || edge_fn(a.xy, b.xy, c.xy) <= 0.f;
		let bmin = min3(a, b, c);
		let bmax = max3(a, b, c);
		let prec = 1.f / 256.f;
		culled = culled || (round(bmin.x - prec) == round(bmax.x) 
			            || round(bmin.y) == round(bmax.y + prec));
		let in_front_of_z = a.z > 0.f && b.z > 0.f && c.z > 0.f;
		culled = culled && in_front_of_z;

		visbuffer[gtid].culled = culled;
	}
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
// https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
[shader("compute")]
[numthreads(128, 1, 1)]
void sw(u32 gid: SV_GroupID, u32 gtid: SV_GroupIndex) {
	let init = Init(gid, gtid, false);
	let vert_count = init.meshlet.vertex_count;
	let tri_count = init.meshlet.tri_count;
	let dim = f32x2(Constants.output.size());

	if (gtid < vert_count) {
		let v = init.transform(gtid).uv;
		Pos[gtid] = f32x4(v.xy * dim, v.z, 1.f);
	}
	workgroup_sync_barrier();

	if (gtid >= tri_count)
		return;

	let t = init.tri(gtid);
	let v0 = Pos[t.x].xyz;
	let v1 = Pos[t.y].xyz;
	let v2 = Pos[t.z].xyz;
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
