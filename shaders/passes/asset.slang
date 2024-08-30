module asset;

import graph;

public struct Vertex {
	public f32x3 position;
	public f32x3 normal;
	public f32x2 uv;
}

public struct Aabb {
	public f32x3 center;
	public f32x3 half_extent;
}

public struct BvhNode {
	public Aabb aabb;
	public f32x4 lod_bounds;
	public f32 parent_error;
	public u32 children_offset;
	public u32 child_count;
}

public struct Instance {
	public f32 transform[12];
	public u8* mesh;
	public Aabb aabb;

	public f32x4x4 get_transform() {
		let t = this.transform;
		// clang-format off
		return {
			t[0], t[3], t[6], t[9], 
			t[1], t[4], t[7], t[10], 
			t[2], t[5], t[8], t[11], 
			0.f, 0.f, 0.f, 1.f,
		};
		// clang-format on
	}

	public BvhNode node(u32 offset) {
		return *(BvhNode*)(this.mesh + offset);
	}

	public Meshlet meshlet(u32 offset) {
		return *(Meshlet*)(this.mesh + offset);
	}
}

public struct Meshlet {
	public Aabb aabb;
	public f32x4 lod_bounds;
	public f32 error;
	public u32 vertex_offset;
	public u32 index_offset;
	public u8 vertex_count;
	public u8 tri_count;
	u16 _pad;

	public Vertex vertex(u8* mesh, u32 id) {
		return ((Vertex*)(mesh + this.vertex_offset))[id];
	}

	public u32x3 tri(u8* mesh, u32 id) {
		let i = ((u32*)(mesh + this.index_offset))[id];
		return u32x3(u8(i >> 0), u8(i >> 8), u8(i >> 16));
	}
}

public struct Camera {
	public f32x4x4 view;
	public f32x4x4 view_proj;
	public f32 w;
	public f32 h;
	public f32 near;
	f32 _pad;
	public f32x4 frustum;
}
