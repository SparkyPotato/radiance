module visbuffer;

import graph;
import passes.asset;
import visbuffer_extra;
import cull;

extern struct Debug : VisbufferDebug = Null;

public struct VisBufferData {
	public u32 meshlet_id;
	public u32 triangle_id;

	public u32 encode() {
		return (this.meshlet_id << 7) | this.triangle_id;
	}

	public __init(u32 data) {
		VisBufferData ret;
		ret.meshlet_id = data >> 7;
		ret.triangle_id = data & 0x7f;
		return ret;
	}
}

public struct VisBuffer {
	public f32 depth;
	public u32 data;

	public u64 encode() {
		return (u64(asuint(this.depth)) << 32) | u64(this.data);
	}

	public __init(u64 data) {
		VisBuffer ret;
		ret.depth = asfloat(u32(data >> 32));
		ret.data = u32(data & 0xffffffff);
		return ret;
	}
}

public struct VisBufferTex {
	STex<u64, D2> tex;
	Debug dbg;

	public void write(u32x2 pos, f32 depth, u32 data, u32 mode) {
		VisBuffer d = { depth, data };
		this.tex.atomic_max(pos, d.encode());
		this.dbg.write(pos, depth, mode);
	}

	public u32x2 pixel_of_uv(f32x2 uv) {
		return this.tex.pixel_of_uv(uv);
	}

	public VisBuffer decode(f32x2 uv) {
		return this.decode(this.pixel_of_uv(uv));
	}

	public VisBuffer decode(u32x2 p) {
		return VisBuffer(this.tex.load(p));
	}

	public VisBufferData data(f32x2 uv) {
		return VisBufferData(this.decode(uv).data);
	}

	public u32 overdraw(f32x2 uv) {
		return this.dbg.overdraw(uv);
	}

	public u32 hwsw(f32x2 uv) {
		return this.dbg.hwsw(uv);
	}

	public u32x2 size() {
		return this.tex.size();
	}
};

public struct VertexTransform {
	public f32x4 clip;
	public f32x3 uv;

	public __init(f32x4x4 mvp, Vertex vertex) {
		this.clip = mul(mvp, f32x4(vertex.position, 1.f));
		let ndc = clip.xyz / clip.w;
		let uv = ndc.xy * f32x2(0.5f, -0.5f) + 0.5f;
		this.uv = f32x3(uv, ndc.z);
	}
};

// http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
public struct DecodedTri {
	public Vertex v0;
	public Vertex v1;
	public Vertex v2;
	public f32x3 lambda;
	public f32x3 ddx;
	public f32x3 ddy;

	public __init(Instance* instances, Camera cam, VisBufferData data, f32x2 uv, f32 depth, u32x2 size, NodePointer p) {
		let instance = instances[p.instance];
		let meshlet = instance.meshlet(p.node_offset);
		let mvp = mul(cam.view_proj, instance.get_transform());

		let t = meshlet.tri(instance.mesh, data.triangle_id);
		this.v0 = meshlet.vertex(instance.mesh, t.x);
		this.v1 = meshlet.vertex(instance.mesh, t.y);
		this.v2 = meshlet.vertex(instance.mesh, t.z);
		let v0 = VertexTransform(mvp, this.v0).clip;
		let v1 = VertexTransform(mvp, this.v1).clip;
		let v2 = VertexTransform(mvp, this.v2).clip;

		let inv_w = 1.f / f32x3(v0.w, v1.w, v2.w);
		let ndc0 = v0.xy * inv_w.x;
		let ndc1 = v1.xy * inv_w.y;
		let ndc2 = v2.xy * inv_w.z;

		let inv_det = 1.f / (determinant(f32x2x2(ndc2 - ndc1, ndc0 - ndc1)));
		this.ddx = f32x3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * inv_det * inv_w;
		this.ddy = f32x3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * inv_det * inv_w;
		var ddx_sum = dot(this.ddx, 1.f);
		var ddy_sum = dot(this.ddy, 1.f);

		let p_ndc = (uv - 0.5f) * f32x2(2.f, -2.f);
		let delta = p_ndc - ndc0;
		let persp_inv_w = inv_w.x + delta.x * ddx_sum + delta.y * ddy_sum;
		let persp_w = 1.f / persp_inv_w;

		this.lambda.x = persp_w * (inv_w.x + delta.x * this.ddx.x + delta.y * this.ddy.x);
		this.lambda.y = persp_w * (delta.x * this.ddx.y + delta.y * this.ddy.y);
		this.lambda.z = persp_w * (delta.x * this.ddx.z + delta.y * this.ddy.z);

		this.ddx *= 2.f / size.x;
		this.ddy *= -2.f / size.y;
		ddx_sum *= 2.f / size.x;
		ddy_sum *= -2.f / size.y;

		let persp_ddx = 1.f / (persp_inv_w + ddx_sum);
		let persp_ddy = 1.f / (persp_inv_w + ddy_sum);

		this.ddx = persp_ddx * (this.lambda * persp_inv_w + this.ddx) - this.lambda;
		this.ddy = persp_ddy * (this.lambda * persp_inv_w + this.ddy) - this.lambda;
	}

	public f32x3 normal() {
		return this.interp(this.v0.normal, this.v1.normal, this.v2.normal);
	}

	public T interp<T : IFloat>(T a, T b, T c) {
		return a * T(this.lambda.x) + b * T(this.lambda.y) + c * T(this.lambda.z);
	}
}
