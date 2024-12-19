module brdf;

import graph;
import graph.util.rng;

public struct BrdfSample {
	public f32x3 L;
	public f32x3 wi;
	public f32 pdf;
}

public interface Brdf {
	f32x3 eval(f32x3 wo, f32x3 wi);

	BrdfSample sample(inout Rng rng, f32x3 wo);
}

// Resources:
// PBRT of course: https://pbr-book.org/4ed/contents
// [0] Self Shadow: https://blog.selfshadow.com/links/
// [1] Cook Torrance: https://graphics.pixar.com/library/ReflectanceModel/paper.pdf
// [2] Disney PBR:
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] GGX: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
// [4] Filament: https://google.github.io/filament/Filament.md.html#materialsystem
// [5] Eric Heitz' Smith extension: https://jcgt.org/published/0003/02/03/paper.pdf
// [6] Naty Hoffman Notes:
// https://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf [7] Implenting
// Disney BRDF: https://schuttejoe.github.io/post/disneybsdf/ [8] Karis reference:
// https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

public struct Lambertian : Brdf {
	public f32x3 base_color;

	public __init(f32x3 base_color) {
		this.base_color = base_color;
	}

	public f32x3 eval(f32x3 wo, f32x3 wi) {
		return this.base_color / PI;
	}

	public BrdfSample sample(inout Rng rng, f32x3 wo) {
		let wi = rng.sample_cos_hemi();
		let pdf = wi.z / PI;
		let L = this.eval(wo, wi);
		return { L, wi, pdf };
	}
}

