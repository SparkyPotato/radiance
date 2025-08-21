module light_tree_build_interface;

import graph;

import light_tree;

public struct TempBvhNode {
	public NodeData data;
	// LSB is if is inner node.
	public u32 L;
	// LSB is if is inner node.
	public u32 R;
}

public interface LightTreeBuildInterface {
	public NodeData load_leaf(u32 index);
	public f32x3 load_leaf_center(u32 index);
	public void store_node(u32 index, LightTreeNode node, NodeData merged);
}
