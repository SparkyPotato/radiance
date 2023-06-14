#include "radiance-core/interface.l.hlsl"

struct Cone {
    // 0, 1, 2: axis
    // 3: cutoff
    u32 data;
};

struct Meshlet {
	float3x4 transform;
	u32 start_index;
	u32 start_vertex;
	u16 tri_and_vert_count;
	Cone cone;
	u16 _pad;
};

struct Command {
    u32 index_count;
    u32 instance_count;
    u32 first_index;
    i32 vertex_offset;
    // Use instance ID to tell the shader which meshlet to use.
    u32 first_instance;
};

struct PushConstants {
    Buf<Meshlet> meshlets;
    Buf<Command> commands;
    // 0: draw count
    Buf<u32> util;
    u32 meshlet_count;
};

PUSH PushConstants Constants;

[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    u32 index = id.x;
    if (index >= Constants.meshlet_count) {
        return;
    }

    Meshlet meshlet = Constants.meshlets.load(index);
    u32 out_index = Constants.util.atomic_add(0, 1);

    Command command;

    u32 tri_count = meshlet.tri_and_vert_count >> 8;
    command.index_count = tri_count * 3;
    command.instance_count = 1;
    command.first_index = meshlet.start_index;
    command.vertex_offset = i32(meshlet.start_vertex);
    command.first_instance = index;

    Constants.commands.store(out_index, command);
}
