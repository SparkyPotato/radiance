import graph;
import graph.util;
import asset;
import passes.mesh.cull;
import passes.mesh.visbuffer;

enum DebugVis : u32 {
	Triangles,
	Meshlets,
	Overdraw,
	HwSw,
	Normals,
}

struct PushConstants {
	Instance* instances;
	Camera* camera;
	VisBufferReader read;
	u32* highlighted;
	u32 highlight_count;
	u32 vis;
	u32 bottom;
	u32 top;
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

f32x3 heatmap(f32 heat) {
	f32x3 cols[] = {
		f32x3(0.f, 0.f, 0.f), f32x3(0.f, 0.f, 1.f), f32x3(0.f, 1.f, 1.f), f32x3(0.f, 1.f, 0.f),
		f32x3(1.f, 1.f, 0.f), f32x3(1.f, 0.f, 0.f), f32x3(1.f, 1.f, 1.f),
	};
	let scaled = heat * (cols.getCount() - 1.f);
	let bot = u32(floor(scaled));
	let top = u32(ceil(scaled));
	return lerp(cols[bot], cols[top], scaled - f32(bot));
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
		if (let p = Constants.read.decode(clamp(pos + samples[i], 0, size))) {
			d = p.depth;
		}
		let depth = log2(Constants.camera[0].near / d + 1.f);
		grad_x += sobel_x[i] * depth;
		grad_y += sobel_y[i] * depth;
	}
	let ma = max(abs(grad_x), abs(grad_y)) * 2.f;
	if (is_instance_highlighted(instance)) {
		return lerp(col, f32x3(1.f, 0.6f, 0.f), saturate(ma * 10.f));
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
			let value = Constants.read.overdraw(input.uv);
			let heat = clamp((f32(value) - f32(Constants.bottom)) / f32(Constants.top - Constants.bottom), 0.f, 1.f);
			col = heatmap(heat);
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
			col = abs(tri.normal());
			break;
		}
	}

	col = sobel(input.uv, col, p.meshlet.instance);
	return f32x4(col, 1.f);
}
