#define PI 3.14159265359f

struct LightingData {
	// Normalized vector from the shading point towards the camera.
	float3 view;
	// Normalized vector from the shading point towards the light source.
	float3 light;
	// Normalized normal to the surface.
	float3 normal;
	// Normalized halfway between `normal` and `light`.
	float3 half;
	// `dot(normal, view)`
	f32 n_v;
	// `dot(normal, light)`
	f32 n_l;
	// `dot(normal, half)`
	f32 n_h;
	// `dot(light, half)`
	f32 l_h;

	static LightingData calculate(float3 view, float3 light, float3 normal) {
		float3 half = normalize(view + light);
		f32 n_v = abs(dot(normal, view)) + 1e-5;
		f32 n_l = clamp(dot(normal, light), 0.f, 1.f);
		f32 n_h = clamp(dot(normal, half), 0.f, 1.f);
		f32 l_h = clamp(dot(light, half), 0.f, 1.f);

		LightingData ret = { view, light, normal, half, n_v, n_l, n_h, l_h };
		return ret;
	}
};

struct MatInput {
	float4 base_color;
	float3 normal;
	float3 emissive;
	f32 metallic;
	// Perceptual roughness squared.
	f32 alpha;
};

// Resources:
// PBRT of course: https://pbr-book.org/4ed/contents
// [0] Self Shadow: https://blog.selfshadow.com/links/
// [1] Cook Torrance: https://graphics.pixar.com/library/ReflectanceModel/paper.pdf
// [2] Disney PBR: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] GGX: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
// [4] Filament: https://google.github.io/filament/Filament.md.html#materialsystem
// [5] Eric Heitz' Smith extension: https://jcgt.org/published/0003/02/03/paper.pdf 
// [6] Naty Hoffman Notes: https://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
// [7] Implenting Disney BRDF: https://schuttejoe.github.io/post/disneybsdf/
// [8] Karis reference: https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

// GGX (Trowbridge-Reitz) normal distribution. [3]
// Basically everyone uses this.
f32 D_GGX(LightingData l, MatInput m) {
	f32 a2 = m.alpha * m.alpha;
	f32 d = ((l.n_h * a2 - l.n_h) * l.n_h + 1.f);
	return a2 / (d * d * PI);
}

// Height-correlated Smith Visibility Function. [5]
// Where V = G / 4 * n_l * n_v
// Filament says Heitz' height correlated Smith is the correct and exact G term. [4]
// I agree with them.
f32 V_GGX(LightingData l, MatInput m) {
	f32 a2 = m.alpha * m.alpha;
	f32 vf = l.n_l * sqrt(l.n_v * l.n_v * (1.f - a2) + a2);
	f32 lf = l.n_v * sqrt(l.n_l * l.n_l * (1.f - a2) + a2);
	// Possible optimization:
	// f32 v = l.n_l * (l.n_v * (1.f - m.alpha) + m.alpha);
	// f32 l = l.n_v * (l.n_l * (1.f - m.alpha) + m.alpha);
	return 0.5f / (vf + lf);
}

// The Schlick fresnel.
float3 F_Schlick(f32 u, float3 f0, float3 f90 = 1.f) {
	return f0 + (f90 - f0) * pow(clamp(1.f - u, 0.f, 1.f), 5.f);
}

// The Burley Diffuse BRDF. [2]
float3 BRDF_Burley(LightingData l, MatInput m) {
	f32 f90 = 0.5f + 2.f * m.alpha * l.l_h * l.l_h;
	f32 lf = F_Schlick(l.n_l, 1.f, f90).x;
	f32 vf = F_Schlick(l.n_v, 1.f, f90).x;
	float3 diff_color = (1.f - m.metallic) * m.base_color.rgb;
	return diff_color * lf * vf / PI;
}

// The Cook Torrance Specular BRDF. [1]
// TODO: This is not energy conserving - look at Filament section 4.7. [4]
float3 BRDF_CookTorrance(LightingData l, MatInput m) {
	f32 D = D_GGX(l, m);
	float3 dielectric = 0.04f * (1.f - m.metallic);
	float3 metal = m.base_color.rgb * m.metallic;
	float3 f0 = dielectric + metal; // TODO: Get from material.
	f32 f90 = 1.f;
	float3 F = F_Schlick(l.l_h, f0, f90);
	f32 V = V_GGX(l, m);
	return D * V * F;
}

