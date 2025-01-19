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
	public Aabb aabbs[8];
	public f32x4 lod_bounds[8];
	public f32 parent_errors[8];
	public u32 child_offsets[8];
	public u8 child_counts[8];
}

public struct Transform {
	public f32x3 translation;
	public f32x4 rotation;
	public f32x3 scale;

	public f32x4x4 trans_mat() {
		let x = this.translation.x;
		let y = this.translation.y;
		let z = this.translation.z;
		// clang-format off
		return {
			1.f, 0.f, 0.f, x  ,
			0.f, 1.f, 0.f, y  ,
			0.f, 0.f, 1.f, z  ,
			0.f, 0.f, 0.f, 1.f
		};
		// clang-format on
	}

	public f32x4x4 rot_mat() {
		let x = this.rotation.x;
		let y = this.rotation.y;
		let z = this.rotation.z;
		let w = this.rotation.w;
		let x2 = x * x;
		let y2 = y * y;
		let z2 = z * z;

		// clang-format off
		return {
			1.f - 2.f * (y2 + z2), 2.f * (x * y - z * w), 2.f * (x * z + y * w), 0.f,
			2.f * (x * y + z * w), 1.f - 2.f * (x2 + z2), 2.f * (y * z - x * w), 0.f,
			2.f * (x * z - y * w), 2.f * (y * z + x * w), 1.f - 2.f * (x2 + y2), 0.f,
			0.f                  , 0.f                  , 0.f                  , 1.f
		};
		// clang-format on
	}

	public f32x4x4 scale_mat() {
		let x = this.scale.x;
		let y = this.scale.y;
		let z = this.scale.z;
		// clang-format off
		return {
			x  , 0.f, 0.f, 0.f,
			0.f, y  , 0.f, 0.f,
			0.f, 0.f, z  , 0.f,
			0.f, 0.f, 0.f, 1.f
		};
		// clang-format on
	}

	public f32x4x4 mat() {
		return mul(this.trans_mat(), mul(this.rot_mat(), this.scale_mat()));
	}

	public f32x4x4 inv_mat() {
		var inverted = this;
		inverted.translation *= -1.f;
		inverted.rotation = f32x4(-inverted.rotation.xyz, inverted.rotation.w);
		inverted.scale = 1.f / inverted.scale;
		return mul(inverted.scale_mat(), mul(inverted.rot_mat(), inverted.trans_mat()));
	}
}

public struct Material<U : Uniformity = Uniform> {
	public OTex2D<f32x4, U> base_color;
	public f32x4 base_color_factor;
	public OTex2D<f32x4, U> metallic_roughness;
	public f32 metallic_factor;
	public f32 roughness_factor;
	public OTex2D<f32x4, U> normal;
	public OTex2D<f32x4, U> emissive;
	public f32x3 emissive_factor;
}

public struct Instance<U : Uniformity = Uniform> {
	public Transform transform;
	public Transform last_updated_transform;
	public Aabb aabb;
	public u64 update_frame;
	public u8* mesh;
	public Material<U>* material;

	public BvhNode node(u32 offset) {
		return *(BvhNode*)(this.mesh + offset);
	}

	public Meshlet meshlet(u32 offset) {
		return *(Meshlet*)(this.mesh + offset);
	}

	public Transform prev_transform(u64 frame) {
		if (this.update_frame == frame) {
			return this.last_updated_transform;
		} else {
			return this.transform;
		}
	}
}

public struct RtInstance<U : Uniformity = Uniform> {
	public Transform transform;
	public Vertex* raw_mesh;
	public u32 raw_vertex_count;
	public u32 raw_tri_count;
	public Material<U>* material;
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
	public f32 max_edge_length;

	public Vertex vertex(u8* mesh, u32 id) {
		return ((Vertex*)(mesh + this.vertex_offset))[id];
	}

	public u32x3 tri(u8* mesh, u32 id) {
		let i = mesh + this.index_offset + id * 3;
		return u32x3(i[0], i[1], i[2]);
	}
}

public enum LightType {
	Point,
	Directional,
	Emissive,
}

public struct Light {
	public LightType ty;
	public f32x3 radiance;	  // x value is bitcasted u32 if emissive.
	public f32x3 pos_or_dir;  // pos for point, dir for directional.
}

public struct Camera {
	public Transform transform;
	public f32 w;
	public f32 h;
	public f32 near;

	public f32x4x4 proj() {
		// clang-format off
		return {
			w,   0.f, 0.f, 0.f,
			0.f, 0.f, h,   0.f,
			0.f, 0.f, 0.f, near,
			0.f, 1.f, 0.f, 0.f,
		};
		// clang-format on
	}

	public f32x4x4 view() {
		return this.transform.inv_mat();
	}

	public f32x4x4 inv_proj() {
		// clang-format off
		return {
			1.f / w, 0.f,     0.f,        0.f,
			0.f,     0.f,     0.f,        1.f,
			0.f,     1.f / h, 0.f,        0.f,
			0.f,     0.f,     1.f / near, 0.f,
		};
		// clang-format on
	}

	public f32x4x4 inv_view() {
		return this.transform.mat();
	}

	public f32x4x4 view_proj() {
		return mul(this.proj(), this.view());
	}

	public f32x4x4 inv_view_proj() {
		return mul(this.inv_view(), this.inv_proj());
	}
}
