module debug;

import graph;
import graph.util;
import asset;
import passes.mesh.cull;
import passes.visbuffer;

enum DebugVis : u32 {
	Triangles,
	Meshlets,
	Overdraw,
	HwSw,
	Normals,
	Uvs,
	Error,
	BaseColor,
	Roughness,
	Metallic,
	Emissive,
}

struct PushConstants {
	Instance* instances;
	Camera* camera;
	VisBufferReader read;
	u32* highlighted;
	u32 highlight_count;
	u32 vis;
	f32 overdraw_scale;
};

[vk::push_constant]
PushConstants Constants;

// Stolen from https://gist.github.com/badboy/6267743 and niagara.
// Thanks copilot.
u32 hash(u32 a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

bool is_instance_highlighted(u32 id) {
	var left = 0;
	var right = Constants.highlight_count;
	while (left < right) {
		let m = (left + right) >> 1;
		if (Constants.highlighted[m] < id) {
			left = m + 1;
		} else if (Constants.highlighted[m] > id) {
			right = m;
		} else {
			return true;
		}
	}
	return false;
}

// https://www.shadertoy.com/view/3lBXR3
f32x3 inferno(f32 t) {
	t = saturate(t);
	let c0 = f32x3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
	let c1 = f32x3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
	let c2 = f32x3(11.60249308247187, -3.972853965665698, -15.9423941062914);
	let c3 = f32x3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
	let c4 = f32x3(77.162935699427, -33.40235894210092, -81.80730925738993);
	let c5 = f32x3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
	let c6 = f32x3(25.13112622477341, -12.24266895238567, -23.07032500287172);

	return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

f32x3 sobel(f32x2 uv, f32x3 col, u32 instance) {
	// clang-format off
	i32 sobel_x[] = {
		1, 0, -1,
		2, 0, -2,
		1, 0, -1,
	};
	i32 sobel_y[] = {
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1,
	};
	i32x2 samples[] = {
		i32x2(-1, -1), i32x2(0, -1), i32x2(1, -1),
		i32x2(-1,  0), i32x2(0,  0), i32x2(1,  0),
		i32x2(-1,  1), i32x2(0,  1), i32x2(1,  1)
	};
	// clang-format on

	var grad_x = 0.f;
	var grad_y = 0.f;
	let pos = Constants.read.pixel_of_uv(uv);
	let size = Constants.read.size() - 1;
	for (int i = 0; i < samples.getCount(); i++) {
		var d = 1.f;
		var ins = instance;
		if (let p = Constants.read.decode(clamp(pos + samples[i], 0, size))) {
			d = p.depth;
			ins = p.meshlet.instance;
		}
		let depth = log2(Constants.camera[0].near / d + 1.f);
		grad_x += sobel_x[i] * (depth + select(ins == instance, 0.f, 0.3f));
		grad_y += sobel_y[i] * (depth + select(ins == instance, 0.f, 0.3f));
	}
	let ma = max(abs(grad_x), abs(grad_y)) * 2.f;

	if (is_instance_highlighted(instance)) {
		return lerp(col, f32x3(1.f, 0.6f, 0.f), saturate(ma * 4.f));
	} else {
		return saturate(col * (1.f - ma));
	}
}

[shader("pixel")]
f32x4 main(ScreenOutput input) : SV_Target0 {
	let pix = Constants.read.decode(input.uv);
	if (pix == none)
		return f32x4(0.f, 0.f, 0.f, 1.f);
	let p = pix.value;

	let tri = DecodedTri(Constants.instances, Constants.camera[0], input.uv, Constants.read.size(), p);
	var col = f32x3(1.f, 0.f, 1.f);
	switch (Constants.vis) {
		case DebugVis.Triangles: {
			let h = hash(p.raw.triangle_id);
			col = f32x3(f32(h & 255), f32((h >> 8) & 255), f32((h >> 16) & 255)) / 255.f;
			break;
		}
		case DebugVis.Meshlets: {
			let h = hash(p.meshlet.instance) ^ hash(p.meshlet.node_offset);
			col = f32x3(f32(h & 255), f32((h >> 8) & 255), f32((h >> 16) & 255)) / 255.f;
			break;
		}
		case DebugVis.Overdraw: {
			let value = f32(Constants.read.overdraw(input.uv));
			let heat = 1.f - exp(-value * Constants.overdraw_scale);
			col = inferno(heat);
			break;
		}
		case DebugVis.HwSw: {
			let data = Constants.read.hwsw(input.uv);
			f32x3 cols[] = { f32x3(0.f, 0.f, 0.f), f32x3(0.f, 1.f, 0.f), f32x3(1.f, 1.f, 0.f), f32x3(1.f, 0.f, 0.f) };
			if (data >= cols.getCount())
				col = f32x3(1.f, 0.f, 1.f);
			else
				col = cols[data];
			break;
		}
		case DebugVis.Normals: {
			let basis = tri.tbn_basis();
			let norm = tri.instance->material->normal.get();
			let z = f32x4(0.f, 0.f, 1.f, 0.f);
			let n = norm.load(norm.pixel_of_uv(tri.uv()), z).xyz;
			col = abs(mul(basis, n));
			break;
		}
		case DebugVis.Uvs: {
			col = f32x3(tri.uv(), 0.f);
			break;
		}
		case DebugVis.Error: {
			let res = Constants.read.size();
			let cull = Cull(Constants.camera, tri.instance, 0, res, Tex2D<f32>(), Sampler<Uniform>());
			let error_over_dist = cull.error_over_dist(tri.meshlet->lod_bounds, tri.meshlet->error);
			let thresh = cull.threshold_for_pix(1.f);
			let error = error_over_dist / thresh;

			let count = f32x2(25.f, 40.f);
			let start = floor(input.uv * count) / count;
			let size = f32x2(4.f, 5.f) / res;
			let h = hash(p.meshlet.instance) ^ hash(p.meshlet.node_offset);
			col = f32x3(f32(h & 255), f32((h >> 8) & 255), f32((h >> 16) & 255)) / 255.f;
			let digit = print(input.uv, start, size, error, 0.f, 8.f);
			col = select(digit > 0.5f, 0.f, col);
			break;
		}
		case DebugVis.BaseColor: {
			let mat = tri.instance.material;
			let bc = mat->base_color.get();
			let white = f32x4(1.f, 1.f, 1.f, 1.f);
			col = (bc.load(bc.pixel_of_uv(tri.uv()), white) * mat->base_color_factor).xyz;
			break;
		}
		case DebugVis.Roughness: {
			let mat = tri.instance.material;
			let mr = mat->metallic_roughness.get();
			let r = mr.load(mr.pixel_of_uv(tri.uv()), 1.f).y * mat->roughness_factor;
			col = f32x3(r, r, r);
			break;
		}
		case DebugVis.Metallic: {
			let mat = tri.instance.material;
			let mr = mat->metallic_roughness.get();
			let m = mr.load(mr.pixel_of_uv(tri.uv()), 1.f).z * mat->metallic_factor;
			col = f32x3(m, m, m);
			break;
		}
		case DebugVis.Emissive: {
			let mat = tri.instance.material;
			let bc = mat->emissive.get();
			let white = f32x4(1.f, 1.f, 1.f, 1.f);
			col = bc.load(bc.pixel_of_uv(tri.uv()), white).xyz * mat->emissive_factor;
			break;
		}
	}

	col = sobel(input.uv, col, p.meshlet.instance);
	return f32x4(col, 1.f);
}
