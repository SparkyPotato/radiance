module light_tree_build_tlas;

import graph;

import asset;
import light_tree;
import light_tree_build_interface;

export struct Interface : LightTreeBuildInterface = LightTreeTlasInterface;

struct BlasInstance {
	Transform transform;
	LightBlas* blas;
}

public struct LightTreeTlasInterface : LightTreeBuildInterface {
	LightTreeNode* tree_nodes;
	BlasInstance* root_data;

	public NodeData load_leaf(u32 index) {
		let instance = this.root_data[index];
		return NodeData(instance.blas->data, instance.transform);
	}

	public f32x3 load_leaf_center(u32 index) {
		let data = this.load_leaf(index);
		return (data.aabb_min + data.aabb_max) * 0.5f;
	}

	public void store_node(u32 index, TempBvhNode data) {
		this.tree_nodes[index] = data.to_light_tree();
	}
}
