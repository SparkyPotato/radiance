#include "radiance-core/interface.l.hlsl"

struct Vertex {
	u16 position[3];
	u16 normal[3];
	u16 uv[2];
};

struct Meshlet {
	f32 transform[12];
	u32 start_index;
	u32 start_vertex;
	// 0, 1, 2: axis
    // 3: cutoff
	u32 cone;
	u16 tri_and_vert_count;
	u16 _pad;
};

struct Camera {
    float4x4 view_proj;
};
