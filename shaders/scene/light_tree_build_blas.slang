module light_tree_build_blas;

import graph;

import asset;
import light_tree;
import light_tree_build_interface;

export struct Interface : LightTreeBuildInterface = LightTreeBlasInterface;

public struct LightTreeBlasInterface : LightTreeBuildInterface {
	LightTreeNode* tree_nodes;
	Vertex* vertices;
	u32x3* indices;
	Material* material;
	LightBlas* root_data;

	public NodeData load_leaf(u32 index) {
		let v = this.load_triangle(index);
		return NodeData(v, this.material);
	}

	public f32x3 load_leaf_center(u32 index) {
		let v = this.load_triangle(index);
		return (v[0] + v[1] + v[2]) / 3.f;
	}

	public void store_node(u32 index, TempBvhNode data) {
		this.tree_nodes[index] = data.to_light_tree();
		if (index == 0)
			*this.root_data = LightBlas(this.tree_nodes, this.vertices, this.indices, this.material, data.data);
	}

	f32x3[3] load_triangle(u32 prim) {
		f32x3 v[3];
		let tri = this.indices[prim];
		v[0] = this.vertices[tri.x].position;
		v[1] = this.vertices[tri.y].position;
		v[2] = this.vertices[tri.z].position;
		return v;
	}
}
