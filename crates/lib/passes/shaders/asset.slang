import graph.types;

public struct Vertex {
    public f32x3 position;
    public f32x3 normal;
    public f32x2 uv;
}

public struct MeshletPointer {
    public u32 instance;
    public u32 meshlet;
}

public struct Instance {
    f32 transform[12];
    public u8* mesh;

    public func get_transform() -> float4x4 {
        f32 t[12] = this.transform;
        float4x4 ret = {
            t[0], t[3], t[6], t[9],
            t[1], t[4], t[7], t[10],
            t[2], t[5], t[8], t[11],
            0.f,  0.f,  0.f,  1.f,
        };
        return ret;
    }
}

public struct Meshlet {
    public u32 vertex_offset;
    public u32 index_offset;
    public u8 vertex_count;
    public u8 triangle_count;
    public u16 _pad;
    public f32x4 bounding;
    public f32x4 group_error;
    public f32x4 parent_error;
}

public struct Material {
    public f32 base_color_factor[4];
    public f32 metallic_factor;
    public f32 roughness_factor;
    public f32 emissive_factor[3];
}

public struct Camera {
    public float4x4 view;
    public float4x4 view_proj;
    public f32 w;
    public f32 h;
    public f32 near;
    public f32 _pad[13];
}
