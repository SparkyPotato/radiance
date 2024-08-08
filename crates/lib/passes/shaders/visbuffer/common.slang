import graph.types;
import graph.texture;
import graph.sampler;
import asset;

public struct PushConstants {
    public Instance* instances;
    public MeshletPointer* meshlet_pointers;
    public Camera* camera;
    public Sampler hzb_sampler;
    public Tex<f32, D2> hzb;
    public u32* culled;
    public u32 meshlet_count;
    public u32 width;
    public u32 height;
};

public [vk::push_constant] PushConstants Constants;

public struct PointerWithId {
    public MeshletPointer pointer;
    public u32 id;
};

public struct MeshPayload {
    public PointerWithId pointers[64];
};

public struct VertexOutput {
    public f32x4 position: SV_Position;
};

public struct PrimitiveOutput {
    public [vk::location(0)] u32 data: VisBuffer;
};
