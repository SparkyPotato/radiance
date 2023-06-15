#include "radiance-core/interface.l.hlsl"

struct Vertex {
	u16 position[3];
	u16 normal[3];
	u16 uv[2];
};

struct Cone {
    // 0, 1, 2: axis
    // 3: cutoff
    u32 data;
};

struct Meshlet {
	f32 transform[12];
	u32 start_index;
	u32 start_vertex;
	u16 tri_and_vert_count;
	Cone cone;
	u16 _pad;
};

struct Camera {
    float4x4 view_proj;
};
