#include "radiance-core/interface.l.hlsl"

struct Vertex {
	u16 position[3];
	u16 normal[3];
	u16 uv[2];
};

/// A pointer to a instance and the meshlet within that instance.
struct MeshletPointer {
    u32 instance;
    u32 meshlet;
};

/// An instance of a mesh.
struct Instance {
    f32 transform[12];
    u32 base_meshlet;
};

struct Cone {
    u32 apex;
    u32 axis_cutoff;
};

/// A meshlet is a collection of 124 triangles and 64 vertices.
struct Meshlet {
	f32 aabb_min[3];
	f32 aabb_extent[3];
	Cone cone;
};

struct Camera {
    float4x4 view;
    float4x4 proj;
};
