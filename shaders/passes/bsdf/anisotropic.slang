module anisotropic;

import graph;
import graph.util.rng;
import passes.bsdf.common;

public struct Aniso {
	public f32 ax;
	public f32 ay;

	public __init(f32 roughness, f32 anisotropic) {
		let aspect = sqrt(1.f - 0.9f * anisotropic);
		let alpha = roughness * roughness;
		this.ax = max(0.001f, alpha / aspect);
		this.ay = max(0.001f, alpha * aspect);
	}
}
