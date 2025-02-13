// https://bruop.github.io/exposure/

import graph;
import graph.util.color;

struct PushConstants {
	u32* histogram;
	Tex2D<f32x4> input;
	f32 min_exp;
	f32 inv_exp_range;
};

[vk::push_constant]
PushConstants Constants;

u8 bin_lum(f32 lum) {
	if (lum < 0.001f)
		return 0;

	let exp = log2(lum) + 3.f;	// log2(lum * 8)
	let log = clamp((exp - Constants.min_exp) * Constants.inv_exp_range, 0.0f, 1.0f);
	return u8(log * 254.f + 1.f);
}

groupshared u32 shared_histogram[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void histogram(u32x2 id: SV_DispatchThreadID, u32 gid: SV_GroupIndex, u32x2 group: SV_GroupID) {
	shared_histogram[gid] = 0;
	workgroup_sync_barrier();

	let size = Constants.input.size();
	if (id.x < size.x && id.y < size.y) {
		let color = Constants.input.load(id).xyz;
		let bin = u32(bin_lum(luminance(color)));
		atomic_add(shared_histogram[bin], 1, Scope.Workgroup, Loc.Workgroup);
	}
	workgroup_sync_barrier();

	atomic_add(Constants.histogram[gid], shared_histogram[gid]);
}
