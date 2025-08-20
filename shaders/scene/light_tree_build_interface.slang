module light_tree_build_interface;

import graph;

import light_tree;

public struct TempBvhNode {
	public NodeData data;
	// LSB is if is inner node.
	public u32 L;
	// LSB is if is inner node.
	public u32 R;

	public __init(u32 left, u32 right, NodeData data) {
		this.data = data;
		this.L = left;
		this.R = right;
	}

	public LightTreeNode to_light_tree() {
		LightTreeNode ret;
		ret.light.position = this.data.position.xyz;
		ret.light.variance = this.data.position.w;
		let sg = average_direction_to_sg_axis(this.data.normal_average);
		ret.light.intensity = power_to_intensity(this.data.power, sg.w);
		ret.light.axis = sg.xyz * sg.w;
		ret.L = this.L;
		ret.R = this.R;
		return ret;
	}
}

public interface LightTreeBuildInterface {
	public NodeData load_leaf(u32 index);
	public f32x3 load_leaf_center(u32 index);
	public void store_node(u32 index, TempBvhNode node);
}
