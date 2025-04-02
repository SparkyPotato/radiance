module visbuffer;

import graph;
import asset;
import visbuffer_extra;
import passes.mesh.cull;

extern struct Debug : VisbufferDebug = Null;

public struct VisBufferData {
	public u32 meshlet_id;
	public u32 triangle_id;

	public u32 encode() {
		return (this.meshlet_id << 7) | this.triangle_id;
	}

	public __init(u32 meshlet_id, u32 triangle_id) {
		VisBufferData ret;
		ret.meshlet_id = meshlet_id;
		ret.triangle_id = triangle_id;
		return ret;
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

	public __init(f32 depth, u32 data) {
		VisBuffer ret;
		ret.depth = depth;
		ret.data = data;
		return ret;
	}

	public __init(u64 data) {
		VisBuffer ret;
		ret.depth = asfloat(u32(data >> 32));
		ret.data = u32(data & 0xffffffff);
		return ret;
	}
}

public struct VisBufferTex {
	STex2D<u64, r64u> tex;
	Debug dbg;

	public void write(u32x2 pos, f32 depth, u32 data, u32 mode) {
		let d = VisBuffer(depth, data);
		atomic_max(this.tex[pos], d.encode());
		this.dbg.write(pos, depth, mode);
	}

	public u32x2 pixel_of_uv(f32x2 uv) {
		return this.tex.pixel_of_uv(uv);
	}

	public VisBuffer decode(f32x2 uv) {
		return this.decode(this.pixel_of_uv(uv));
	}

	public VisBuffer decode(u32x2 p) {
		return VisBuffer(this.tex[p]);
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

public struct VisBufferPixel {
	public f32 depth;
	public VisBufferData raw;
	public NodePointer meshlet;
}

public struct VisBufferReader {
	MeshletQueue queue;
	VisBufferTex read;
	u32 _pad;

	public u32x2 pixel_of_uv(f32x2 uv) {
		return this.read.pixel_of_uv(uv);
	}

	public u32x2 size() {
		return this.read.size();
	}

	public Optional<VisBufferPixel> decode(u32x2 pix) {
		let visbuf = this.read.decode(pix);
		if (visbuf.data == 0xffffffff)
			return none;
		VisBufferPixel ret;
		ret.depth = visbuf.depth;
		ret.raw = VisBufferData(visbuf.data);
		ret.meshlet = this.queue.get(ret.raw.meshlet_id);
		return ret;
	}

	public Optional<VisBufferPixel> decode(f32x2 uv) {
		return this.decode(this.pixel_of_uv(uv));
	}

	public u32 overdraw(f32x2 uv) {
		return this.read.overdraw(uv);
	}

	public u32 hwsw(f32x2 uv) {
		return this.read.hwsw(uv);
	}
}

// http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
// TODO: all in model space
public struct DecodedTri {
	public Instance* instance;
	public Meshlet* meshlet;
	public Vertex v0;
	public Vertex v1;
	public Vertex v2;
	public f32x3 lambda;
	public f32x3 ddx;
	public f32x3 ddy;

	public __init(Instance* instances, Camera cam, f32x2 uv, u32x2 size, VisBufferPixel p) {
		this.instance = &instances[p.meshlet.instance];
		this.meshlet = this.instance->meshlet(p.meshlet.node_offset);
		let mvp = mul(cam.view_proj(), this.instance->transform.mat());

		let t = this.meshlet->tri(this.instance.mesh, p.raw.triangle_id);
		this.v0 = this.meshlet->vertex(this.instance.mesh, t.x);
		this.v1 = this.meshlet->vertex(this.instance.mesh, t.y);
		this.v2 = this.meshlet->vertex(this.instance.mesh, t.z);
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

	// https://www.jeremyong.com/graphics/2023/12/16/surface-gradient-bump-mapping
	public f32x3x3 tbn_basis() {
		let normal = this.normal();
		let pos_ddx = this.position_ddx();
		let proj_pos_ddx = pos_ddx - dot(pos_ddx, normal) * normal;
		let pos_ddy = this.position_ddy();
		let proj_pos_ddy = pos_ddy - dot(pos_ddy, normal) * normal;
		let uv_ddx = this.uv_ddx();
		let uv_ddy = this.uv_ddy();

		let jacobian = sign(determinant(f32x2x2(uv_ddx, uv_ddy)));
		var tangent = jacobian * (uv_ddy.y * proj_pos_ddx - uv_ddx.y * proj_pos_ddy);
		if (jacobian != 0.f) {
			tangent = normalize(tangent);
		}
		let bitangent = jacobian * sign(dot(pos_ddy, cross(normal, pos_ddx))) * cross(normal, tangent);

		return transpose(f32x3x3(tangent, bitangent, normal));
	}

	public f32x3 position() {
		return this.interp(this.v0.position, this.v1.position, this.v2.position);
	}

	public f32x3 position_ddx() {
		return this.ddx_of(this.v0.position, this.v1.position, this.v2.position);
	}

	public f32x3 position_ddy() {
		return this.ddy_of(this.v0.position, this.v1.position, this.v2.position);
	}

	public f32x2 uv() {
		return this.interp(this.v0.uv, this.v1.uv, this.v2.uv);
	}

	public f32x2 uv_ddx() {
		return this.ddx_of(this.v0.uv, this.v1.uv, this.v2.uv);
	}

	public f32x2 uv_ddy() {
		return this.ddy_of(this.v0.uv, this.v1.uv, this.v2.uv);
	}

	public T interp<T : IFloat>(T a, T b, T c) {
		return a * T(this.lambda.x) + b * T(this.lambda.y) + c * T(this.lambda.z);
	}

	public T ddx_of<T : IFloat>(T a, T b, T c) {
		return a * T(this.ddx.x) + b * T(this.ddx.y) + c * T(this.ddx.z);
	}

	public T ddy_of<T : IFloat>(T a, T b, T c) {
		return a * T(this.ddy.x) + b * T(this.ddy.y) + c * T(this.ddy.z);
	}
}
