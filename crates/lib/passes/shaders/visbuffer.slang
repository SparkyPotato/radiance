import graph.types;

public struct VisBufferData {
    public u32 meshlet_pointer_id;
    public u32 triangle_id;

	public __init(u32 data) {
		VisBufferData result;
        result.meshlet_pointer_id = (data >> 7) - 1;
        result.triangle_id = data & 0x7f;
        return result;
	}

    public func encode() -> u32 {
        return ((meshlet_pointer_id + 1) << 7) | triangle_id;
    }
};
