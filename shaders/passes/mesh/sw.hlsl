struct PushConstants {
	uint64_t instances;
	uint64_t camera;
	uint64_t early_hw;
	uint64_t early_sw;
	uint64_t late_hw;
	uint64_t late_sw;
	uint32_t output;
#ifdef DEBUG
	uint32_t overdraw;
	uint32_t hwsw;
#endif
};

[[vk::binding(1, 0)]] RWTexture2D<uint64_t> O64[];
[[vk::binding(1, 0)]] RWTexture2D<uint32_t> O32[];
[[vk::push_constant]] PushConstants Constants;

struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
};

struct Aabb {
	float3 center;
	float3 half_extent;
};

struct Meshlet {
	Aabb aabb;
	float4 lod_bounds;
	float error;
	uint vertex_offset;
	uint index_offset;
	uint16_t vert_and_tri_count;
	uint16_t _pad;
	float max_edge_length;

	uint vertex_count() {
		return (this.vert_and_tri_count >> 0) & 0xff;
	}

	uint tri_count() {
		return (this.vert_and_tri_count >> 8) & 0xff;
	}

	Vertex vertex(uint64_t mesh, uint id) {
		return vk::RawBufferLoad<Vertex>(mesh + this.vertex_offset + sizeof(Vertex) * id);
	}

	uint3 tri(uint64_t mesh, uint id) {
		uint x = vk::RawBufferLoad<uint>(mesh + this.index_offset + id * 3, 1);	
		return uint3((x >> 0) & 0xff, (x >> 8) & 0xff, (x >> 16) & 0xff);
	}
};

struct Transform {
	float3 translation;
	float4 rotation;
	float3 scale;

	float4x4 trans_mat() {
		float x = this.translation.x;
		float y = this.translation.y;
		float z = this.translation.z;
		// clang-format off
		float4x4 ret = {
			1.f, 0.f, 0.f, x  ,
			0.f, 1.f, 0.f, y  ,
			0.f, 0.f, 1.f, z  ,
			0.f, 0.f, 0.f, 1.f
		};
		// clang-format on
		return ret;
	}

	float4x4 rot_mat() {
		float x = this.rotation.x;
		float y = this.rotation.y;
		float z = this.rotation.z;
		float w = this.rotation.w;
		float x2 = x * x;
		float y2 = y * y;
		float z2 = z * z;

		// clang-format off
		float4x4 ret = {
			1.f - 2.f * (y2 + z2), 2.f * (x * y - z * w), 2.f * (x * z + y * w), 0.f,
			2.f * (x * y + z * w), 1.f - 2.f * (x2 + z2), 2.f * (y * z - x * w), 0.f,
			2.f * (x * z - y * w), 2.f * (y * z + x * w), 1.f - 2.f * (x2 + y2), 0.f,
			0.f                  , 0.f                  , 0.f                  , 1.f
		};
		// clang-format on
		return ret;
	}

	float4x4 scale_mat() {
		float x = this.scale.x;
		float y = this.scale.y;
		float z = this.scale.z;
		// clang-format off
		float4x4 ret = {
			x  , 0.f, 0.f, 0.f,
			0.f, y  , 0.f, 0.f,
			0.f, 0.f, z  , 0.f,
			0.f, 0.f, 0.f, 1.f
		};
		// clang-format on
		return ret;
	}

	float4x4 mat() {
		return mul(this.trans_mat(), mul(this.rot_mat(), this.scale_mat()));
	}
};

struct Instance {
	Transform transform;
	uint64_t mesh;
	Aabb aabb;

	float4x4 get_transform() {
		return this.transform.mat();
	}

	Meshlet meshlet(uint offset) {
		return vk::RawBufferLoad<Meshlet>(this.mesh + offset);
	}
};

struct Camera {
	float4x4 view;
	float4x4 view_proj;
	float h;
	float near;
};

struct NodePointer {
	uint instance;
	uint node_offset;
};

groupshared float3 Pos[128];

uint queue_len(uint64_t queue) {
	return vk::RawBufferLoad<uint>(queue);
}

NodePointer queue_get(uint64_t queue, uint id) {
	return vk::RawBufferLoad<NodePointer>(queue + sizeof(uint3) + id * sizeof(NodePointer));
}

uint2 output_size() {
	uint w, h;
	O64[Constants.output].GetDimensions(w, h);
	return uint2(w, h);
}

float edge_fn(float2 a, float2 b, float2 c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

void write(uint2 pos, float depth, uint data, uint mode) {
	uint64_t visbuffer = (uint64_t(asuint(depth)) << 32) | uint64_t(data);
	InterlockedMax(O64[Constants.output][pos], visbuffer);
#ifdef DEBUG
	uint64_t mask = (uint64_t(asuint(depth)) << 32) | uint64_t(mode);
	InterlockedMax(O64[Constants.hwsw][pos], mask);
	InterlockedAdd(O32[Constants.overdraw][pos], 1);
#endif
}

template<typename T>
T min3(T a, T b, T c) {
	return min(a, min(b, c));
}

template<typename T>
T max3(T a, T b, T c) {
	return max(a, max(b, c));
}

struct Tri {
	int2 minv;
	int2 maxv;
	float3 w_x;
	float3 w_y;
	float3 w_row;
	float z_x;
	float z_y;
	float z_row;
	uint write;
};

void scanline(Tri t) {
	float3 e012 = -t.w_x;
	bool3 oe = e012 < 0.f;
	float3 ie012 = select(e012 != 0.f, 1.f / e012, 1e8);
	int width = t.maxv.x - t.minv.x;
	for (int y = t.minv.y; y <= t.maxv.y; y++) {
		float3 cross_x = t.w_row * ie012;
		float3 min_x2 = select(oe, cross_x, 0.f);
		float3 max_x2 = select(oe, width, cross_x);
		uint x0 = uint(ceil(max3(min_x2.x, min_x2.y, min_x2.z)));
		uint x1 = uint(min3(max_x2.x, max_x2.y, max_x2.z));
		float3 w = t.w_row + t.w_x * float(x0);
		float z = t.z_row + t.z_x * float(x0);
		x0 += t.minv.x;
		x1 += t.minv.x;
		for (int x = x0; x <= x1; x++) {
			if (min3(w.x, w.y, w.z) >= 0.f)
				write(uint2(x, y), z, t.write, 1);
			w += t.w_x;
			z += t.z_x;
		}
		t.w_row += t.w_y;
		t.z_row += t.z_y;
	}
}

void bounding(Tri t) {
	for (int y = t.minv.y; y <= t.maxv.y; y++) {
		float3 w = t.w_row;
		float z = t.z_row;
		for (int x = t.minv.x; x <= t.maxv.x; x++) {
			if (min3(w.x, w.y, w.z) >= 0.f)
				write(uint2(x, y), z, t.write, 2);
			w += t.w_x;
			z += t.z_x;
		}
		t.w_row += t.w_y;
		t.z_row += t.z_y;
	}
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
// https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
[numthreads(128, 1, 1)]
void main(uint gid: SV_GroupID, uint gtid: SV_GroupIndex) {
#ifdef EARLY
	NodePointer p = queue_get(Constants.early_sw, gid);
	uint mid = queue_len(Constants.early_hw) + gid;
#else
	NodePointer p = queue_get(Constants.late_sw, gid);
	uint mid = queue_len(Constants.early_hw) + queue_len(Constants.early_sw) + queue_len(Constants.late_hw) + gid;
#endif
	Instance instance = vk::RawBufferLoad<Instance>(Constants.instances + sizeof(Instance) * p.instance);
	uint64_t mesh = instance.mesh;
	Meshlet meshlet = instance.meshlet(p.node_offset);
	Camera cam = vk::RawBufferLoad<Camera>(Constants.camera);
	float4x4 mvp = mul(cam.view_proj, instance.get_transform());
	uint2 dim = output_size();

	if (gtid < meshlet.vertex_count()) {
		Vertex v = meshlet.vertex(mesh, gtid);
		float4 clip = mul(mvp, float4(v.position, 1.f));
		float3 ndc = clip.xyz / clip.w;
		float2 uv = ndc.xy * float2(0.5f, -0.5f) + 0.5f;
		Pos[gtid] = float3(uv * dim, ndc.z);
	}
	GroupMemoryBarrierWithGroupSync();

	if (gtid >= meshlet.tri_count())
		return;

	uint3 t = meshlet.tri(mesh, gtid);
	float3 v0 = Pos[t.x];
	float3 v1 = Pos[t.y];
	float3 v2 = Pos[t.z];
	uint write = (mid << 7) | gtid;

	float3 mi = min3(v0, v1, v2);
	float3 ma = max3(v0, v1, v2);
	int2 minv = int2(floor(mi.xy));
	int2 maxv = int2(floor(ma.xy));
	minv = max(minv, int2(0, 0));
	maxv = min(maxv, int2(dim - 1));
	maxv = min(maxv, minv + 31); // Try not to TDR
	if (any(minv > maxv))
		return;

	float3 w_x = float3(v1.y - v2.y, v2.y - v0.y, v0.y - v1.y);
	float3 w_y = float3(v2.x - v1.x, v0.x - v2.x, v1.x - v0.x);
	float par_area = edge_fn(v0.xy, v1.xy, v2.xy);
	if (par_area < 0.f)
		return;
	float3 v_z = float3(v0.z, v1.z, v2.z) / par_area;
	float z_x = dot(v_z, w_x);
	float z_y = dot(v_z, w_y);

	float2 start = minv + 0.5f;
	// saturates for top left rule.
	float3 w_row = float3(edge_fn(v1.xy, v2.xy, start) /* - saturate(w_x.x + saturate(1.f + w_y.x)) */,
					      edge_fn(v2.xy, v0.xy, start) /* - saturate(w_x.y + saturate(1.f + w_y.y)) */,
					      edge_fn(v0.xy, v1.xy, start) /* - saturate(w_x.z + saturate(1.f + w_y.z)) */);
	float z_row = dot(v_z, w_row);
	Tri tri = { minv, maxv, w_x, w_y, w_row, z_x, z_y, z_row, write };
	if (WaveActiveAnyTrue(maxv.x - minv.x > 4)) {
		scanline(tri);
	} else {
		bounding(tri);
	}
}
