#pragma once

#include "radiance-graph/interface.l.hlsl"

struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
};

struct Aabb {
	float3 center;
	float3 half_extent;

	float3 get_corner(u32 id) {
		f32 x = this.half_extent.x;
		f32 y = this.half_extent.y;
		f32 z = this.half_extent.z;
		x = ((id >> 0) & 1) == 0 ? x : -x;
		y = ((id >> 1) & 1) == 0 ? y : -y;
		z = ((id >> 2) & 1) == 0 ? z : -z;
		return this.center + float3(x, y, z);
	}
};

struct BvhNodePointer {
	u32 instance;
	u32 node;
};

struct MeshletPointer {
	u32 instance;
	u32 meshlet_offset;
};

struct BvhNode {
	Aabb aabb;
	float4 lod_bounds;
	f32 parent_error;
	u32 children_offset;
	u32 child_count;
};

struct Instance {
	f32 transform[12];
	Buf<bytes> mesh;
	Aabb aabb;

	float4x4 get_transform() {
		f32 t[12] = this.transform;
		float4x4 ret = {
			t[0], t[3], t[6], t[9],
			t[1], t[4], t[7], t[10],
			t[2], t[5], t[8], t[11],
			0.f,  0.f,  0.f,  1.f,
		};
		return ret;
	}
};

struct Meshlet {
	Aabb aabb;
	float4 lod_bounds;
	f32 error;
	u32 vertex_offset;
	u32 index_offset;
	u16 vert_and_tri_count;
	u16 _pad;

	u32 vertex_count() {
		return this.vert_and_tri_count & 0xff;
	}

	u32 tri_count() {
		return (this.vert_and_tri_count >> 8) & 0xff;
	}

	Vertex vertex(Buf<bytes> mesh, u32 vertex) {
		return mesh.load<Vertex>(this.vertex_offset, vertex);
	}

	u32 tri(Buf<bytes> mesh, u32 tri) {
		return mesh.load<u32>(this.index_offset, tri);
	}
};

struct Material {
	float4 base_color_factor;
	f32 metallic_factor;
	f32 roughness_factor;
	float3 emissive_factor;
};

struct Camera {
	float4x4 view;
	float4x4 view_proj;
	f32 w;
	f32 h;
	f32 near;
	f32 _pad;
	float4 frustum;
};
