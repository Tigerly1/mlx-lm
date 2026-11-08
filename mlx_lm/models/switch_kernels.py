# Copyright © 2026 Apple Inc.

"""Metal kernels for quantized SwitchGLU at small batch (decode, batched decode, short verify steps).

For a few tokens per step, the generic path reads each (token, expert) slot's weights separately. Here:
- T = 1: one simdgroup per expert row computes gate and up together, and the down projection is split over
  (slot, row block) so all k experts are in flight at once.
- T > 1: slots are grouped by expert into tiles of up to 8 and multiplied with simdgroup MMA, so each
  expert's weights are read and dequantized once per tile instead of once per slot.

Supported: affine 4-bit, group size 64, no bias, bf16/fp16 activations, SwiGLU or tanh-GELU GLU.
The output has the same per-slot shape as SwitchGLU, so model routing code is unchanged. Gradients use the
gather_qmm path (custom_function vjp), so training through SwitchGLU keeps working.
"""

from functools import partial

import mlx.core as mx
import mlx.nn as nn

HEADER = """
// GLU activation on the gate (ACT 0 = SiLU as mlx-lm swiglu, 1 = tanh-GELU as nn.gelu_approx, Gemma-4 geglu)
template <int A> inline float glu_act(float g) {
    if (A == 1) return 0.5f * g * (1.f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    return g / (1.f + metal::exp(-g));
}
// load 32 activations, prescale by 1/16^(i%8), return sum(x)
template <typename T>
inline float load_x32(const device T* xp, thread float* xs) {
    float s = 0.f;
    for (uint j = 0; j < 4; ++j) {
        float sc = 1.f;
        for (uint i = 0; i < 8; ++i) {
            float v = float(xp[j * 8 + i]);
            s += v;
            xs[j * 8 + i] = v * sc;
            sc *= 0.0625f;
        }
    }
    return s;
}
inline float dot_q4x32(thread const float* xs, uint4 w) {
    float acc = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 8; ++i) acc += xs[j * 8 + i] * float(word & (0xFu << (4 * i)));
    }
    return acc;
}
// two tokens share one weight decode: and + cvt once, two fma
inline void dot2_q4x32(thread const float* xa, thread const float* xb, uint4 w, thread float& da, thread float& db) {
    float a = 0.f, b = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 8; ++i) { float q = float(word & (0xFu << (4 * i))); a += xa[j * 8 + i] * q; b += xb[j * 8 + i] * q; }
    }
    da += a; db += b;
}
// A-fragment pair for MMA step s: P interleaves two words (low 16 bits: word a nibbles, high: word b).
// Raw codes plus an exact offset: (q | 0x6400) as half = 1024 + q, (q | 0x4300) as bfloat = 128 + q.
// Scale and bias are applied in fp32 per scale group (bias corrected by -OFF * scale).
template <typename M> inline vec<M, 2> ar_qo2(uint P, uint s);
template <> inline half2 ar_qo2<half>(uint P, uint s) { return as_type<half2>(((P >> (4 * s)) & 0x000F000Fu) | 0x64006400u); }
template <> inline vec<bfloat, 2> ar_qo2<bfloat>(uint P, uint s) { return as_type<vec<bfloat, 2>>(((P >> (4 * s)) & 0x000F000Fu) | 0x43004300u); }
// per-token sums of one 64-wide activation group for this lane's fragment cols (fn, fn+1):
// lane-local sum over its 8 values, then over the 8 fragment rows (lane bits 1, 2, 4).
template <typename T>
inline float2 ar_colsum(vec<T, 4> a0, vec<T, 4> a1, vec<T, 4> b0, vec<T, 4> b1) {
    float2 v = float2(0.f);
    for (uint i = 0; i < 4; ++i) { v.x += float(a0[i]) + float(a1[i]); v.y += float(b0[i]) + float(b1[i]); }
    v += simd_shuffle_xor(v, 2); v += simd_shuffle_xor(v, 4); v += simd_shuffle_xor(v, 16);
    return v;
}
// read the tile built by the map kernel: n slots (0 = no tile here), ids into sl[0..n)
inline uint ar_tile(const device uint* tl, uint p, thread uint* sl, uint VT) {
    const device uint* t = tl + p * (VT + 1);
    uint n = t[0];
    for (uint i = 0; i < VT; ++i) sl[i] = (i < n) ? t[1 + i] : 0u;
    return n;
}
"""

GEMV_GATE_UP = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint slot = thread_position_in_grid.y;
    if (row0 >= F) return;
    constexpr uint NW = D / 8, NG = D / GS;
    uint e = idx[slot];
    const device T* xr = x + (slot / K) * D;
    size_t r0 = (size_t)e * F + row0;
    float ag[R] = {0.f}, au[R] = {0.f};
    float xs[32];
    for (uint c = lane; c < NW / 4; c += 32) {
        float xsum = load_x32(xr + c * 32, xs);
        uint grp = (c * 32) / GS;
        for (uint r = 0; r < R; ++r) {
            size_t rr = r0 + r;
            float dg = dot_q4x32(xs, ((const device uint4*)(gq + rr * NW))[c]);
            float du = dot_q4x32(xs, ((const device uint4*)(uq + rr * NW))[c]);
            ag[r] += float(gs[rr * NG + grp]) * dg + float(gb[rr * NG + grp]) * xsum;
            au[r] += float(us[rr * NG + grp]) * du + float(ub[rr * NG + grp]) * xsum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float g = simd_sum(ag[r]), u = simd_sum(au[r]);
        if (lane == 0) h[slot * F + row0 + r] = T(glu_act<ACT>(g) * u);
    }
"""

GEMV_DOWN = """
    // threadgroup = (32, K): simdgroup j computes the rows of slot t*K+j and writes y[slot].
    constexpr uint RB = R * 32 / LPR;  // rows per simdgroup
    uint lane = thread_index_in_simdgroup;
    uint sub = lane % LPR;
    uint j = thread_position_in_threadgroup.y;
    uint rb0 = (thread_position_in_grid.x / 32) * RB;
    uint row0 = rb0 + (lane / LPR) * R;
    uint t = thread_position_in_grid.y / K;
    uint slot = t * K + j;
    bool ok = row0 < D;
    constexpr uint NW = F / 8, NG = F / GS;
    size_t r0 = (size_t)idx[slot] * D + (ok ? row0 : 0);
    float acc[R] = {0.f};
    float hs[32];
    for (uint c = sub; ok && c < NW / 4; c += LPR) {
        float hsum = load_x32(h + slot * F + c * 32, hs);
        uint grp = (c * 32) / GS;
        for (uint r = 0; r < R; ++r) {
            size_t rr = r0 + r;
            float d = dot_q4x32(hs, ((const device uint4*)(dq + rr * NW))[c]);
            acc[r] += float(ds[rr * NG + grp]) * d + float(db[rr * NG + grp]) * hsum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float a = acc[r];
        for (uint o = LPR / 2; o > 0; o >>= 1) a += simd_shuffle_xor(a, o);
        if (sub == 0 && ok) y[slot * D + row0 + r] = T(a);
    }
"""

MAP = """
    uint p = thread_position_in_grid.x;
    if (p >= TK) return;
    int e = idx[p];
    uint rank = 0;
    for (uint q = 0; q < p; ++q) rank += (idx[q] == e) ? 1u : 0u;
    device uint* t = tl + p * (VT + 1);
    if (rank % VT != 0) { t[0] = 0; return; }
    uint n = 0;
    for (uint q = p; q < TK && n < VT; ++q) if (idx[q] == e) t[1 + n++] = q;
    t[0] = n;
"""

TILE_GATE_UP = """
    uint lane = thread_index_in_simdgroup;
    uint ks = simdgroup_index_in_threadgroup;   // K split: SGK simdgroups share 8 rows
    uint row0 = threadgroup_position_in_grid.x * 8;
    uint p = thread_position_in_grid.y;
    threadgroup float red[SGK * 128];
    const device uint* tt = tl + p * (VT + 1);
    uint n = tt[0];
    if (n == 0 || row0 >= NR) return;
    size_t ex = idx[tt[1]];
    uint rw0 = row0 + ks * RS;  // small tiles: v31 per-slot GEMV, RS = 8 / SGK rows per simdgroup
    if (n == 2 && NSMALL >= 2) {   // 2-slot tile: scalar GEMV, weights decoded once for both tokens
        constexpr uint NWs = D / 8, NGs = D / GS;
        uint sa = tt[1], sb = tt[2];
        const device T* xra = x + (sa / K) * D;
        const device T* xrb = x + (sb / K) * D;
        float ga[RS] = {0.f}, gb_[RS] = {0.f}, ua[RS] = {0.f}, ub_[RS] = {0.f}, xs[32], xt[32];
        for (uint c = lane; c < NWs / 4; c += 32) {
            float xsa = load_x32(xra + c * 32, xs), xsb = load_x32(xrb + c * 32, xt);
            uint grp = (c * 32) / GS;
            for (uint r = 0; r < RS; ++r) {
                size_t q = ex * F + rw0 + r;
                float dga = 0.f, dgb = 0.f, dua = 0.f, dub = 0.f;
                dot2_q4x32(xs, xt, ((const device uint4*)(gq + q * NWs))[c], dga, dgb);
                dot2_q4x32(xs, xt, ((const device uint4*)(uq + q * NWs))[c], dua, dub);
                float sg = float(gs[q * NGs + grp]), bg = float(gb[q * NGs + grp]);
                float su = float(us[q * NGs + grp]), bu = float(ub[q * NGs + grp]);
                ga[r] += sg * dga + bg * xsa; gb_[r] += sg * dgb + bg * xsb;
                ua[r] += su * dua + bu * xsa; ub_[r] += su * dub + bu * xsb;
            }
        }
        for (uint r = 0; r < RS; ++r) {
            float g = simd_sum(ga[r]), u = simd_sum(ua[r]);
            float g2 = simd_sum(gb_[r]), u2 = simd_sum(ub_[r]);
            if (lane == 0) {
                h[sa * F + rw0 + r] = T(glu_act<ACT>(g) * u);
                h[sb * F + rw0 + r] = T(glu_act<ACT>(g2) * u2);
            }
        }
        return;
    }
    if (n == 1) {
        constexpr uint NWs = D / 8, NGs = D / GS;
        for (uint t = 0; t < n; ++t) {
            uint slot = tt[1 + t];
            const device T* xr = x + (slot / K) * D;
            float ag[RS] = {0.f}, au[RS] = {0.f}, xs[32];
            for (uint c = lane; c < NWs / 4; c += 32) {
                float xsum = load_x32(xr + c * 32, xs);
                uint grp = (c * 32) / GS;
                for (uint r = 0; r < RS; ++r) {
                    size_t q = ex * F + rw0 + r;
                    float dg = dot_q4x32(xs, ((const device uint4*)(gq + q * NWs))[c]);
                    float du = dot_q4x32(xs, ((const device uint4*)(uq + q * NWs))[c]);
                    ag[r] += float(gs[q * NGs + grp]) * dg + float(gb[q * NGs + grp]) * xsum;
                    au[r] += float(us[q * NGs + grp]) * du + float(ub[q * NGs + grp]) * xsum;
                }
            }
            for (uint r = 0; r < RS; ++r) {
                float g = simd_sum(ag[r]), u = simd_sum(au[r]);
                if (lane == 0) h[slot * F + rw0 + r] = T(glu_act<ACT>(g) * u);
            }
        }
        return;
    }

    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);   // fragment row of this lane
    uint fn = (qid & 2) * 2 + (lane % 2) * 2; // fragment cols fn, fn+1
    uint s0 = tt[1 + min(fn, n - 1)], s1 = tt[1 + min(fn + 1, n - 1)];
    size_t rr = (size_t)idx[tt[1]] * NR + row0 + fm;
    constexpr uint NW = NK / 8, NB = NK / 64;

    const device T* xa = x + (s0 / K) * D + fm * 8;
    const device T* xb = x + (s1 / K) * D + fm * 8;
    const device uint2* g2 = (const device uint2*)(gq + rr * NW) + fn / 2;
    const device uint2* u2 = (const device uint2*)(uq + rr * NW) + fn / 2;
    simdgroup_matrix<float, 8, 8> Cg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Cu = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Pg, Pu;
    simdgroup_matrix<MT, 8, 8> Ag, Au, B;
    for (uint g = ks; g < NB; g += SGK) {   // one 64-wide scale group: col c, step s <-> k = 8c + s
        uint2 wg = g2[g * 4], wu = u2[g * 4];
        uint pg[2] = {(wg.x & 0xFFFFu) | (wg.y << 16), (wg.x >> 16) | (wg.y & 0xFFFF0000u)};
        uint pu[2] = {(wu.x & 0xFFFFu) | (wu.y << 16), (wu.x >> 16) | (wu.y & 0xFFFF0000u)};
        vec<T, 4> a0 = ((const device vec<T, 4>*)(xa + g * 64))[0], a1 = ((const device vec<T, 4>*)(xa + g * 64))[1];
        vec<T, 4> b0 = ((const device vec<T, 4>*)(xb + g * 64))[0], b1 = ((const device vec<T, 4>*)(xb + g * 64))[1];
        for (uint s = 0; s < 8; ++s) {
            { vec<MT, 2> v_ = ar_qo2<MT>(pg[s / 4], s % 4); Ag.thread_elements()[0] = v_.x; Ag.thread_elements()[1] = v_.y; }
            { vec<MT, 2> v_ = ar_qo2<MT>(pu[s / 4], s % 4); Au.thread_elements()[0] = v_.x; Au.thread_elements()[1] = v_.y; }
            B.thread_elements()[0] = MT(s < 4 ? a0[s] : a1[s - 4]); B.thread_elements()[1] = MT(s < 4 ? b0[s] : b1[s - 4]);
            if (s == 0) { simdgroup_multiply(Pg, Ag, B); simdgroup_multiply(Pu, Au, B); }
            else { simdgroup_multiply_accumulate(Pg, Ag, B, Pg); simdgroup_multiply_accumulate(Pu, Au, B, Pu); }
        }
        float2 sx = ar_colsum(a0, a1, b0, b1);
        uint grp = rr * NB + g;
        float sg_ = float(gs[grp]), su = float(us[grp]);
        float bg = float(gb[grp]) - OFF * sg_, bu = float(ub[grp]) - OFF * su;   // A holds q + OFF
        for (uint i = 0; i < 2; ++i) {
            Cg.thread_elements()[i] = fma(sg_, Pg.thread_elements()[i], fma(bg, sx[i], Cg.thread_elements()[i]));
            Cu.thread_elements()[i] = fma(su, Pu.thread_elements()[i], fma(bu, sx[i], Cu.thread_elements()[i]));
        }
    }
    for (uint i = 0; i < 2; ++i) {
        red[ks * 128 + lane * 2 + i] = Cg.thread_elements()[i];
        red[ks * 128 + 64 + lane * 2 + i] = Cu.thread_elements()[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float g = 0.f, u = 0.f;
        for (uint j = 0; j < SGK; ++j) { g += red[j * 128 + lane * 2 + i]; u += red[j * 128 + 64 + lane * 2 + i]; }
        if (fn + i < n) h[(i ? s1 : s0) * F + row0 + fm] = T(glu_act<ACT>(g) * u);
    }
"""

TILE_DOWN = """
    uint lane = thread_index_in_simdgroup;
    uint ks = simdgroup_index_in_threadgroup;   // K split: SGK simdgroups share 8 rows
    uint row0 = threadgroup_position_in_grid.x * 8;
    uint p = thread_position_in_grid.y;
    threadgroup float red[SGK * 128];
    const device uint* tt = tl + p * (VT + 1);
    uint n = tt[0];
    if (n == 0 || row0 >= NR) return;
    size_t ex = idx[tt[1]];
    uint rw0 = row0 + ks * RS;  // small tiles: v31 per-slot GEMV, RS = 8 / SGK rows per simdgroup
    if (n == 2 && NSMALL >= 2) {   // 2-slot tile: LPR GEMV, weights decoded once for both slots
        constexpr uint NWs = F / 8, NGs = F / GS;
        constexpr uint RR = RS * LPR / 32;
        uint sa = tt[1], sb = tt[2];
        uint sub = lane % LPR, rg = rw0 + (lane / LPR) * RR;
        float aa[RR] = {0.f}, ab[RR] = {0.f}, hs[32], ht[32];
        for (uint c = sub; c < NWs / 4; c += LPR) {
            float hsa = load_x32(h + sa * F + c * 32, hs), hsb = load_x32(h + sb * F + c * 32, ht);
            uint grp = (c * 32) / GS;
            for (uint r = 0; r < RR; ++r) {
                size_t q = ex * D + rg + r;
                float da = 0.f, db_ = 0.f;
                dot2_q4x32(hs, ht, ((const device uint4*)(dq + q * NWs))[c], da, db_);
                float sc = float(ds[q * NGs + grp]), bi = float(db[q * NGs + grp]);
                aa[r] += sc * da + bi * hsa; ab[r] += sc * db_ + bi * hsb;
            }
        }
        float wa = wts[sa], wb = wts[sb];
        for (uint r = 0; r < RR; ++r) {
            float a = aa[r], b = ab[r];
            for (uint o = LPR / 2; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b += simd_shuffle_xor(b, o); }
            if (sub == 0) { y[sa * D + rg + r] = wa * a; y[sb * D + rg + r] = wb * b; }
        }
        return;
    }
    if (n == 1) {
        constexpr uint NWs = F / 8, NGs = F / GS;
        for (uint t = 0; t < n; ++t) {
            uint slot = tt[1 + t];
            // LPR lanes per row (v4 trick): 32 / LPR row groups of RR rows, all lanes busy when F/32 < 32
            constexpr uint RR = RS * LPR / 32;
            uint sub = lane % LPR, rg = rw0 + (lane / LPR) * RR;
            float acc[RR] = {0.f}, hs[32];
            for (uint c = sub; c < NWs / 4; c += LPR) {
                float hsum = load_x32(h + slot * F + c * 32, hs);
                uint grp = (c * 32) / GS;
                for (uint r = 0; r < RR; ++r) {
                    size_t q = ex * D + rg + r;
                    float d = dot_q4x32(hs, ((const device uint4*)(dq + q * NWs))[c]);
                    acc[r] += float(ds[q * NGs + grp]) * d + float(db[q * NGs + grp]) * hsum;
                }
            }
            float w = wts[slot];
            for (uint r = 0; r < RR; ++r) {
                float a = acc[r];
                for (uint o = LPR / 2; o > 0; o >>= 1) a += simd_shuffle_xor(a, o);
                if (sub == 0) y[slot * D + rg + r] = w * a;
            }
        }
        return;
    }

    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);   // fragment row of this lane
    uint fn = (qid & 2) * 2 + (lane % 2) * 2; // fragment cols fn, fn+1
    uint s0 = tt[1 + min(fn, n - 1)], s1 = tt[1 + min(fn + 1, n - 1)];
    size_t rr = (size_t)idx[tt[1]] * NR + row0 + fm;
    constexpr uint NW = NK / 8, NB = NK / 64;

    const device T* ha = h + s0 * F + fm * 8;
    const device T* hb = h + s1 * F + fm * 8;
    const device uint2* d2 = (const device uint2*)(dq + rr * NW) + fn / 2;
    simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> P;
    simdgroup_matrix<MT, 8, 8> A, B;
    for (uint g = ks; g < NB; g += SGK) {
        uint2 w = d2[g * 4];
        uint pw[2] = {(w.x & 0xFFFFu) | (w.y << 16), (w.x >> 16) | (w.y & 0xFFFF0000u)};
        vec<T, 4> a0 = ((const device vec<T, 4>*)(ha + g * 64))[0], a1 = ((const device vec<T, 4>*)(ha + g * 64))[1];
        vec<T, 4> b0 = ((const device vec<T, 4>*)(hb + g * 64))[0], b1 = ((const device vec<T, 4>*)(hb + g * 64))[1];
        for (uint s = 0; s < 8; ++s) {
            { vec<MT, 2> v_ = ar_qo2<MT>(pw[s / 4], s % 4); A.thread_elements()[0] = v_.x; A.thread_elements()[1] = v_.y; }
            B.thread_elements()[0] = MT(s < 4 ? a0[s] : a1[s - 4]); B.thread_elements()[1] = MT(s < 4 ? b0[s] : b1[s - 4]);
            if (s == 0) simdgroup_multiply(P, A, B);
            else simdgroup_multiply_accumulate(P, A, B, P);
        }
        float2 sx = ar_colsum(a0, a1, b0, b1);
        uint grp = rr * NB + g;
        float sc = float(ds[grp]), bi = float(db[grp]) - OFF * sc;
        for (uint i = 0; i < 2; ++i) C.thread_elements()[i] = fma(sc, P.thread_elements()[i], fma(bi, sx[i], C.thread_elements()[i]));
    }
    for (uint i = 0; i < 2; ++i) red[ks * 128 + lane * 2 + i] = C.thread_elements()[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float a = 0.f;
        for (uint j = 0; j < SGK; ++j) a += red[j * 128 + lane * 2 + i];
        uint slot = i ? s1 : s0;
        if (fn + i < n) y[slot * D + row0 + fm] = wts[slot] * a;
    }
"""


R_UP, R_DOWN, SG_PER_TG = (
    1,
    2,
    4,
)  # GEMV rows per simdgroup (gate/up, down), simdgroups per threadgroup
VT, SGK_UP, NSMALL = (
    8,
    8,
    2,
)  # tile width, K-split simdgroups (gate/up), tiles <= NSMALL slots use GEMV


def _make_kernels():
    if not mx.metal.is_available():
        return None
    mk = partial(mx.fast.metal_kernel, header=HEADER)
    w = ["gq", "gs", "gb", "uq", "us", "ub"]
    return {
        "gemv_gate_up": mk(
            name="switch_glu_gemv_gate_up",
            input_names=["x", *w, "idx"],
            output_names=["h"],
            source=GEMV_GATE_UP,
        ),
        "gemv_down": mk(
            name="switch_glu_gemv_down",
            input_names=["h", "dq", "ds", "db", "idx"],
            output_names=["y"],
            source=GEMV_DOWN,
        ),
        "map": mx.fast.metal_kernel(
            name="switch_glu_tile_map",
            input_names=["idx"],
            output_names=["tl"],
            source=MAP,
        ),
        "tile_gate_up": mk(
            name="switch_glu_tile_gate_up",
            input_names=["x", *w, "idx", "tl"],
            output_names=["h"],
            source=TILE_GATE_UP,
        ),
        "tile_down": mk(
            name="switch_glu_tile_down",
            input_names=["h", "dq", "ds", "db", "idx", "wts", "tl"],
            output_names=["y"],
            source=TILE_DOWN,
        ),
    }


_kernels = _make_kernels()


def _lanes_per_row(chunks, iters=3):
    """Smallest power of 2 >= chunks / iters, at most 32."""
    lanes = 1
    while lanes < 32 and lanes * iters < chunks:
        lanes *= 2
    return lanes


def _forward(x, indices, weights, act):
    """x: [T, D], indices: [T, k] -> per-slot expert outputs [T * k, D]."""
    gq, gs, gb, uq, us, ub, dq, ds, db = weights
    T, k = indices.shape
    F, D = gq.shape[1], dq.shape[1]
    w = [gq, gs, gb, uq, us, ub]
    if T == 1:
        base = [
            ("T", x.dtype),
            ("S", gs.dtype),
            ("D", D),
            ("F", F),
            ("K", k),
            ("GS", 64),
            ("ACT", act),
        ]
        h = _kernels["gemv_gate_up"](
            inputs=[x, *w, indices],
            template=base + [("R", R_UP)],
            grid=(32 * F // R_UP, k, 1),
            threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(k, F)],
            output_dtypes=[x.dtype],
        )[0]
        lpr = _lanes_per_row(F // 32)
        rb = R_DOWN * 32 // lpr
        return _kernels["gemv_down"](
            inputs=[h, dq, ds, db, indices],
            template=base + [("R", R_DOWN), ("LPR", lpr)],
            grid=(32 * (-(-D // rb)), k, 1),
            threadgroup=(32, k, 1),
            output_shapes=[(k, D)],
            output_dtypes=[x.dtype],
        )[0]
    # T > 1: every slot is its own row (K = 1); the tile map groups rows by expert
    n = T * k
    x = mx.broadcast_to(x[:, None], (T, k, D)).reshape(n, D)
    indices = indices.reshape(n)
    bf = x.dtype == mx.bfloat16
    tm = [
        ("T", x.dtype),
        ("S", gs.dtype),
        ("D", D),
        ("F", F),
        ("K", 1),
        ("GS", 64),
        ("ACT", act),
        ("VT", VT),
        ("TK", n),
        ("MT", mx.bfloat16 if bf else mx.float16),
        ("OFF", 128 if bf else 1024),
    ]
    tl = _kernels["map"](
        inputs=[indices],
        template=[("VT", VT), ("TK", n)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n * (VT + 1),)],
        output_dtypes=[mx.uint32],
    )[0]
    h = _kernels["tile_gate_up"](
        inputs=[x, *w, indices, tl],
        template=tm
        + [
            ("NR", F),
            ("NK", D),
            ("SGK", SGK_UP),
            ("NSMALL", NSMALL),
            ("RS", 8 // SGK_UP),
        ],
        grid=(32 * SGK_UP * F // 8, n, 1),
        threadgroup=(32 * SGK_UP, 1, 1),
        output_shapes=[(n, F)],
        output_dtypes=[x.dtype],
    )[0]
    lpr = _lanes_per_row(F // 32)
    sgk = max(1, lpr // (4 * R_DOWN))
    lpr = max(lpr, 4 * sgk)
    y = _kernels["tile_down"](
        inputs=[h, dq, ds, db, indices, mx.ones((n,), mx.float32), tl],
        template=tm
        + [
            ("NR", D),
            ("NK", F),
            ("SGK", sgk),
            ("NSMALL", NSMALL),
            ("RS", 8 // sgk),
            ("LPR", lpr),
        ],
        grid=(32 * sgk * D // 8, n, 1),
        threadgroup=(32 * sgk, 1, 1),
        output_shapes=[(n, D)],
        output_dtypes=[mx.float32],
    )[0]
    return y.astype(x.dtype)


def _reference(x, indices, weights, act):
    """Same math through gather_qmm (used for gradients)."""
    gq, gs, gb, uq, us, ub, dq, ds, db = weights
    xx = mx.expand_dims(x, (-2, -3))
    mm = partial(
        mx.gather_qmm, rhs_indices=indices, transpose=True, group_size=64, bits=4
    )
    g, u = mm(xx, gq, gs, gb), mm(xx, uq, us, ub)
    a = nn.silu(g) if act == 0 else nn.gelu_approx(g)
    return mm(a * u, dq, ds, db).squeeze(-2).reshape(-1, x.shape[-1])


def _make_op(act):
    @mx.custom_function
    def op(x, indices, *weights):
        return _forward(x, indices, weights, act)

    @op.vjp
    def op(primals, cotangent, output):
        x, indices, *weights = primals
        _, (dx,) = mx.vjp(
            lambda x: _reference(x, indices, weights, act), [x], [cotangent]
        )
        return (dx, mx.zeros_like(indices), *(mx.zeros_like(w) for w in weights))

    return op


_ops = {0: _make_op(0), 1: _make_op(1)}


def max_load(D, F):
    """Average tokens per expert (T*k/E) up to which these kernels beat gather_qmm (M4 Pro measurement)."""
    return 24 if D * F >= 3 << 19 else 8


def switch_glu(glu, x, indices):
    """Per-slot output of SwitchGLU `glu` (as SwitchGLU.__call__), or None if this path does not apply."""
    if getattr(glu.gate_proj, "mode", None) == "mxfp4":
        return _switch_glu_fp4(glu, x, indices)
    act = getattr(glu.activation, "metal_act", None)
    if _kernels is None or act is None or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    projs = (glu.gate_proj, glu.up_proj, glu.down_proj)
    for p in projs:
        fmt = (
            getattr(p, "bits", None),
            getattr(p, "group_size", None),
            getattr(p, "mode", "affine"),
        )
        if fmt != (4, 64, "affine") or "bias" in p:
            return None
    k = indices.shape[-1]
    T = indices.size // k
    E, F, D = glu.gate_proj.weight.shape[0], glu.gate_proj.weight.shape[1], x.shape[-1]
    if T * k / E > max_load(D, F):
        return None
    weights = [a for p in projs for a in (p.weight, p.scales, p.biases)]
    y = _ops[act](x.reshape(T, D), indices.reshape(T, k).astype(mx.uint32), *weights)
    return y.reshape(*indices.shape, D)


# ---------------------------------------------------------------------------------------------
# MXFP4 experts (gpt-oss): E2M1 weights, E8M0 scales per 32, fp32-added biases, SwiGLU (ACT 0)
# or gpt-oss's clamped SwiGLU (ACT 1: alpha 1.702, limit 7). Per-slot outputs come from the
# pair (T < 8) and group (T >= 8) kernels with unit routing weights; the k-sum stays in the model.
# ---------------------------------------------------------------------------------------------

FP4_HEADER = """
// two E2M1 codes (bits 0..3 and 16..19 of p) -> half2, value * 2^-14 (exact)
inline half2 fp4x2(uint p) {
    return as_type<half2>(((p & 0x00080008u) << 12) | ((p & 0x00070007u) << 9));
}
// E8M0 scale byte -> float 2^(s-127), exact for s = 1..254 (s = 0 -> 0). The decode's 2^-14 is undone once
// per output (swiglu / routing weight * 2^14), not folded in here: (s + 14) << 23 overflows for s >= 241.
inline float e8s(uint s) { return as_type<float>(s << 23); }
inline float swiglu(float g, float u) { g *= 0x1p14f; return g / (1.f + metal::exp(-g)) * (u * 0x1p14f); }
template <typename T>
inline void load_x32(const device T* xp, thread float* xs) {
    for (uint i = 0; i < 32; ++i) xs[i] = float(xp[i]);
}
// dot of 32 fp4 weights (one uint4) with 32 floats. nibble i of word j <-> weight 8j + i.
inline float dot_fp4x32(thread const float* xs, uint4 w) {
    float acc = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 4; ++i) {
            float2 v = float2(fp4x2(word >> (4 * i)));
            acc = fma(xs[j * 8 + i], v.x, acc);
            acc = fma(xs[j * 8 + i + 4], v.y, acc);
        }
    }
    return acc;
}
// two activation vectors share one weight decode
inline void dot2_fp4x32(thread const float* xa, thread const float* xb, uint4 w, thread float& da, thread float& db) {
    float a = 0.f, b = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 4; ++i) {
            float2 v = float2(fp4x2(word >> (4 * i)));
            a = fma(xa[j * 8 + i], v.x, a); a = fma(xa[j * 8 + i + 4], v.y, a);
            b = fma(xb[j * 8 + i], v.x, b); b = fma(xb[j * 8 + i + 4], v.y, b);
        }
    }
    da = a; db = b;
}

template <int A>
inline float act(float g, float u) {
    if (A == 0) return g / (1.f + metal::exp(-g)) * u;
    g = metal::min(g, 7.0f);
    u = metal::clamp(u, -7.0f, 7.0f);
    return g / (1.f + metal::exp(-1.702f * g)) * (u + 1.f);
}
"""

FP4_PAIR_GATE_UP = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint slot = thread_position_in_grid.y;
    if (row0 >= F) return;
    constexpr uint NC = D / 32;
    int e = idx[slot];

    uint rank = 0, nxt = TK;
    for (uint q = lane; q < TK; q += 32) {
        bool same = idx[q] == e;
        rank += (same && q < slot) ? 1u : 0u;
        if (same && q > slot) nxt = min(nxt, q);
    }
    rank = simd_sum(rank); nxt = simd_min(nxt);
    if (rank & 1u) return;

    size_t r0 = (size_t)e * F + row0;
    const device T* xr = x + (slot / K) * D;
    float xs[32];
    if (nxt == TK) {
        float ag[R] = {0.f}, au[R] = {0.f};
        for (uint c = lane; c < NC; c += 32) {
            load_x32(xr + c * 32, xs);
            for (uint r = 0; r < R; ++r) {
                size_t q = (r0 + r) * NC + c;
                ag[r] += e8s(gs[q]) * dot_fp4x32(xs, ((const device uint4*)gq)[q]);
                au[r] += e8s(us[q]) * dot_fp4x32(xs, ((const device uint4*)uq)[q]);
            }
        }
        for (uint r = 0; r < R; ++r) {
            float g = simd_sum(ag[r]), u = simd_sum(au[r]);
            if (lane == 0) h[slot * F + row0 + r] = T(act<ACT>(g * 0x1p14f + bg[r0 + r], u * 0x1p14f + bu[r0 + r]));
        }
        return;
    }
    const device T* xr2 = x + (nxt / K) * D;
    float xt[32];
    float ga[R] = {0.f}, gb[R] = {0.f}, ua[R] = {0.f}, ub[R] = {0.f};
    for (uint c = lane; c < NC; c += 32) {
        load_x32(xr + c * 32, xs);
        load_x32(xr2 + c * 32, xt);
        for (uint r = 0; r < R; ++r) {
            size_t q = (r0 + r) * NC + c;
            float a, b, sg = e8s(gs[q]), su = e8s(us[q]);
            dot2_fp4x32(xs, xt, ((const device uint4*)gq)[q], a, b); ga[r] += sg * a; gb[r] += sg * b;
            dot2_fp4x32(xs, xt, ((const device uint4*)uq)[q], a, b); ua[r] += su * a; ub[r] += su * b;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float g = simd_sum(ga[r]), u = simd_sum(ua[r]), g2 = simd_sum(gb[r]), u2 = simd_sum(ub[r]);
        if (lane == 0) {
            h[slot * F + row0 + r] = T(act<ACT>(g * 0x1p14f + bg[r0 + r], u * 0x1p14f + bu[r0 + r]));
            h[nxt * F + row0 + r] = T(act<ACT>(g2 * 0x1p14f + bg[r0 + r], u2 * 0x1p14f + bu[r0 + r]));
        }
    }
"""

FP4_PAIR_DOWN = """
    constexpr uint RB = R * 32 / LPR;
    uint lane = thread_index_in_simdgroup;
    uint sub = lane % LPR;
    uint row0 = (thread_position_in_grid.x / 32) * RB + (lane / LPR) * R;
    uint slot = thread_position_in_grid.y;
    int e = idx[slot];

    uint rank = 0, nxt = TK;
    for (uint q = lane; q < TK; q += 32) {
        bool same = idx[q] == e;
        rank += (same && q < slot) ? 1u : 0u;
        if (same && q > slot) nxt = min(nxt, q);
    }
    rank = simd_sum(rank); nxt = simd_min(nxt);
    if (rank & 1u) return;

    bool ok = row0 < D;
    constexpr uint NC = F / 32;
    size_t r0 = (size_t)e * D + (ok ? row0 : 0);
    float hs[32], ht[32];
    float aa[R] = {0.f}, ab[R] = {0.f};
    if (nxt == TK) {
        for (uint c = sub; ok && c < NC; c += LPR) {
            load_x32(h + slot * F + c * 32, hs);
            for (uint r = 0; r < R; ++r) {
                size_t q = (r0 + r) * NC + c;
                aa[r] += e8s(ds[q]) * dot_fp4x32(hs, ((const device uint4*)dq)[q]);
            }
        }
    } else {
        for (uint c = sub; ok && c < NC; c += LPR) {
            load_x32(h + slot * F + c * 32, hs);
            load_x32(h + nxt * F + c * 32, ht);
            for (uint r = 0; r < R; ++r) {
                size_t q = (r0 + r) * NC + c;
                float a, b, sd = e8s(ds[q]);
                dot2_fp4x32(hs, ht, ((const device uint4*)dq)[q], a, b);
                aa[r] += sd * a; ab[r] += sd * b;
            }
        }
    }
    float wa = wts[slot] * 0x1p14f, wb = wts[min(nxt, uint(TK - 1))] * 0x1p14f;
    for (uint r = 0; r < R; ++r) {
        float a = aa[r], b = ab[r];
        for (uint o = LPR / 2; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b += simd_shuffle_xor(b, o); }
        if (sub == 0 && ok) {
            yp[slot * D + row0 + r] = wa * (a + bd[r0 + r] * 0x1p-14f);
            if (nxt < TK) yp[nxt * D + row0 + r] = wb * (b + bd[r0 + r] * 0x1p-14f);
        }
    }
"""

FP4_MAP = """
    uint p = thread_position_in_grid.x;
    if (p >= TK) return;
    int e = idx[p];
    uint rank = 0;
    for (uint q = 0; q < p; ++q) rank += (idx[q] == e) ? 1u : 0u;
    device uint* t = tl + p * (G + 1);
    if (rank % G != 0) { t[0] = 0; return; }
    uint n = 0;
    for (uint q = p; q < TK && n < G; ++q) if (idx[q] == e) t[1 + n++] = q;
    t[0] = n;
"""

FP4_GROUP_GATE_UP = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row0 = threadgroup_position_in_grid.x * RT;
    constexpr uint RS = RT / 8, LPR = 32 / RS, SK = 64 / RT;  // scalar: RS rows/simdgroup; MMA: SK-way K split
    uint slot = thread_position_in_grid.y;
    threadgroup float red[8 * 128];
    if (row0 >= NR) return;
    const device uint* tt = tl + slot * (G + 1);
    uint n = tt[0];
    if (n == 0) return;
    int e = idx[slot];
    uint m[G];
    for (uint i = 0; i < G; ++i) m[i] = tt[1 + min(i, n - 1)];
    uint sub = lane % LPR;
    uint row = row0 + sg * RS + lane / LPR;
    uint rbk = sg / SK, ks = sg % SK;
    constexpr uint NC = NK / 32;
    // MMA fragment coordinates
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint s0 = m[min(fn, n - 1)], s1 = m[min(fn + 1, n - 1)];
    uint rowm = row0 + rbk * 8 + fm;
    size_t rf = (size_t)e * NR + rowm;

    size_t rr = (size_t)e * F + row;
    if (n <= 2) {
        const device T* xr = x + (slot / K) * D;
        float xs[32], ga = 0.f, ua = 0.f, gb = 0.f, ub = 0.f;
        if (n == 1) {
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(xr + c * 32, xs);
                size_t q = rr * NC + c;
                ga += e8s(gs[q]) * dot_fp4x32(xs, ((const device uint4*)gq)[q]);
                ua += e8s(us[q]) * dot_fp4x32(xs, ((const device uint4*)uq)[q]);
            }
        } else {
            const device T* xr2 = x + (m[1] / K) * D;
            float xt[32];
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(xr + c * 32, xs);
                load_x32(xr2 + c * 32, xt);
                size_t q = rr * NC + c;
                float a, b, sg_ = e8s(gs[q]), su = e8s(us[q]);
                dot2_fp4x32(xs, xt, ((const device uint4*)gq)[q], a, b); ga += sg_ * a; gb += sg_ * b;
                dot2_fp4x32(xs, xt, ((const device uint4*)uq)[q], a, b); ua += su * a; ub += su * b;
            }
        }
        for (uint o = LPR / 2; o > 0; o >>= 1) {
            ga += simd_shuffle_xor(ga, o); ua += simd_shuffle_xor(ua, o);
            gb += simd_shuffle_xor(gb, o); ub += simd_shuffle_xor(ub, o);
        }
        if (sub == 0) {
            h[slot * F + row] = T(act<ACT>(ga * 0x1p14f + bg[rr], ua * 0x1p14f + bu[rr]));
            if (n == 2) h[m[1] * F + row] = T(act<ACT>(gb * 0x1p14f + bg[rr], ub * 0x1p14f + bu[rr]));
        }
        return;
    }
    const device T* xa = x + (s0 / K) * D + fm * 4;
    const device T* xb = x + (s1 / K) * D + fm * 4;
    const device uint* gw = gq + rf * (NK / 8) + fn / 2;
    const device uint* uw = uq + rf * (NK / 8) + fn / 2;
    simdgroup_matrix<float, 8, 8> Cg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Cu = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Pg, Pu;
    simdgroup_matrix<half, 8, 8> Ag, Au, B;
    for (uint g = ks; g < NC; g += SK) {
        uint wg = gw[g * 4], wu = uw[g * 4];
        float sg_ = e8s(gs[rf * NC + g]), su = e8s(us[rf * NC + g]);
        half4 a0 = half4(*(const device vec<T, 4>*)(xa + g * 32)), b0 = half4(*(const device vec<T, 4>*)(xb + g * 32));
        for (uint s = 0; s < 4; ++s) {
            half2 vg = fp4x2(wg >> (4 * s)), vu = fp4x2(wu >> (4 * s));
            Ag.thread_elements()[0] = vg.x; Ag.thread_elements()[1] = vg.y;
            Au.thread_elements()[0] = vu.x; Au.thread_elements()[1] = vu.y;
            B.thread_elements()[0] = a0[s]; B.thread_elements()[1] = b0[s];
            if (s == 0) { simdgroup_multiply(Pg, Ag, B); simdgroup_multiply(Pu, Au, B); }
            else { simdgroup_multiply_accumulate(Pg, Ag, B, Pg); simdgroup_multiply_accumulate(Pu, Au, B, Pu); }
        }
        for (uint i = 0; i < 2; ++i) {
            Cg.thread_elements()[i] = fma(sg_, Pg.thread_elements()[i], Cg.thread_elements()[i]);
            Cu.thread_elements()[i] = fma(su, Pu.thread_elements()[i], Cu.thread_elements()[i]);
        }
    }
    for (uint i = 0; i < 2; ++i) {
        red[sg * 128 + lane * 2 + i] = Cg.thread_elements()[i];
        red[sg * 128 + 64 + lane * 2 + i] = Cu.thread_elements()[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float g = 0.f, u = 0.f;
        for (uint j = 0; j < SK; ++j) { g += red[(sg + j) * 128 + lane * 2 + i]; u += red[(sg + j) * 128 + 64 + lane * 2 + i]; }
        if (fn + i < n) h[m[fn + i] * F + rowm] = T(act<ACT>(g * 0x1p14f + bg[rf], u * 0x1p14f + bu[rf]));
    }
"""

FP4_GROUP_DOWN = """
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row0 = threadgroup_position_in_grid.x * RT;
    constexpr uint RS = RT / 8, LPR = 32 / RS, SK = 64 / RT;  // scalar: RS rows/simdgroup; MMA: SK-way K split
    uint slot = thread_position_in_grid.y;
    threadgroup float red[8 * 128];
    if (row0 >= NR) return;
    const device uint* tt = tl + slot * (G + 1);
    uint n = tt[0];
    if (n == 0) return;
    int e = idx[slot];
    uint m[G];
    for (uint i = 0; i < G; ++i) m[i] = tt[1 + min(i, n - 1)];
    uint sub = lane % LPR;
    uint row = row0 + sg * RS + lane / LPR;
    uint rbk = sg / SK, ks = sg % SK;
    constexpr uint NC = NK / 32;
    // MMA fragment coordinates
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);
    uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint s0 = m[min(fn, n - 1)], s1 = m[min(fn + 1, n - 1)];
    uint rowm = row0 + rbk * 8 + fm;
    size_t rf = (size_t)e * NR + rowm;

    size_t rr = (size_t)e * D + row;
    if (n <= 2) {
        float hs[32], a = 0.f, b2 = 0.f;
        if (n == 1) {
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(h + slot * F + c * 32, hs);
                size_t q = rr * NC + c;
                a += e8s(ds[q]) * dot_fp4x32(hs, ((const device uint4*)dq)[q]);
            }
        } else {
            float ht[32];
            for (uint c = sub; c < NC; c += LPR) {
                load_x32(h + slot * F + c * 32, hs);
                load_x32(h + m[1] * F + c * 32, ht);
                size_t q = rr * NC + c;
                float da, db, sd = e8s(ds[q]);
                dot2_fp4x32(hs, ht, ((const device uint4*)dq)[q], da, db);
                a += sd * da; b2 += sd * db;
            }
        }
        for (uint o = LPR / 2; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b2 += simd_shuffle_xor(b2, o); }
        if (sub == 0) {
            yp[slot * D + row] = wts[slot] * (0x1p14f * a + bd[rr]);
            if (n == 2) yp[m[1] * D + row] = wts[m[1]] * (0x1p14f * b2 + bd[rr]);
        }
        return;
    }
    const device T* ha = h + s0 * F + fm * 4;
    const device T* hb = h + s1 * F + fm * 4;
    const device uint* dw = dq + rf * (NK / 8) + fn / 2;
    simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> P;
    simdgroup_matrix<half, 8, 8> A, B;
    for (uint g = ks; g < NC; g += SK) {
        uint w = dw[g * 4];
        float sd = e8s(ds[rf * NC + g]);
        half4 a0 = half4(*(const device vec<T, 4>*)(ha + g * 32)), b0 = half4(*(const device vec<T, 4>*)(hb + g * 32));
        for (uint s = 0; s < 4; ++s) {
            half2 v = fp4x2(w >> (4 * s));
            A.thread_elements()[0] = v.x; A.thread_elements()[1] = v.y;
            B.thread_elements()[0] = a0[s]; B.thread_elements()[1] = b0[s];
            if (s == 0) simdgroup_multiply(P, A, B);
            else simdgroup_multiply_accumulate(P, A, B, P);
        }
        for (uint i = 0; i < 2; ++i) C.thread_elements()[i] = fma(sd, P.thread_elements()[i], C.thread_elements()[i]);
    }
    for (uint i = 0; i < 2; ++i) red[sg * 128 + lane * 2 + i] = C.thread_elements()[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float a = 0.f;
        for (uint j = 0; j < SK; ++j) a += red[(sg + j) * 128 + lane * 2 + i];
        if (fn + i < n) yp[m[fn + i] * D + rowm] = wts[m[fn + i]] * (0x1p14f * a + bd[rf]);
    }
"""


FP4_R_UP, FP4_R_DOWN, FP4_SG, FP4_T_GROUP, FP4_G, FP4_RT_UP, FP4_RT_DOWN = (
    1,
    2,
    4,
    8,
    8,
    16,
    32,
)


def _make_fp4_kernels():
    if not mx.metal.is_available():
        return None
    mk = partial(mx.fast.metal_kernel, header=FP4_HEADER)
    w = ["gq", "gs", "uq", "us", "bg", "bu"]
    return {
        "pair_gate_up": mk(
            name="switch_glu_fp4_pair_gate_up",
            input_names=["x", *w, "idx"],
            output_names=["h"],
            source=FP4_PAIR_GATE_UP,
        ),
        "pair_down": mk(
            name="switch_glu_fp4_pair_down",
            input_names=["h", "dq", "ds", "bd", "idx", "wts"],
            output_names=["yp"],
            source=FP4_PAIR_DOWN,
        ),
        "map": mx.fast.metal_kernel(
            name="switch_glu_fp4_map",
            input_names=["idx"],
            output_names=["tl"],
            source=FP4_MAP,
        ),
        "group_gate_up": mk(
            name="switch_glu_fp4_group_gate_up",
            input_names=["x", *w, "idx", "tl"],
            output_names=["h"],
            source=FP4_GROUP_GATE_UP,
        ),
        "group_down": mk(
            name="switch_glu_fp4_group_down",
            input_names=["h", "dq", "ds", "bd", "idx", "wts", "tl"],
            output_names=["yp"],
            source=FP4_GROUP_DOWN,
        ),
    }


_fp4_kernels = _make_fp4_kernels()


def _rows_per_tg(nr, rt):
    while nr % rt:
        rt //= 2
    return rt


def _forward_fp4(x, indices, weights, act):
    """x: [T, D], indices: [T, k] -> per-slot expert outputs [T * k, D] (biases included)."""
    gq, gs, uq, us, dq, ds, bg, bu, bd = weights
    T, k = indices.shape
    F, D = gq.shape[1], dq.shape[1]
    n = T * k
    ones = mx.ones((n,), mx.float32)
    base = [("T", x.dtype), ("D", D), ("F", F), ("K", k), ("ACT", act), ("TK", n)]
    w = [gq, gs, uq, us, bg, bu]
    K = _fp4_kernels
    if T < FP4_T_GROUP:
        h = K["pair_gate_up"](
            inputs=[x, *w, indices],
            template=base + [("R", FP4_R_UP)],
            grid=(32 * F // FP4_R_UP, n, 1),
            threadgroup=(32 * FP4_SG, 1, 1),
            output_shapes=[(n, F)],
            output_dtypes=[x.dtype],
        )[0]
        lpr = _lanes_per_row(F // 32)
        rb = FP4_R_DOWN * 32 // lpr
        y = K["pair_down"](
            inputs=[h, dq, ds, bd, indices, ones],
            template=base + [("R", FP4_R_DOWN), ("LPR", lpr)],
            grid=(32 * (-(-D // rb)), n, 1),
            threadgroup=(32 * FP4_SG, 1, 1),
            output_shapes=[(n, D)],
            output_dtypes=[mx.float32],
        )[0]
    else:
        rt_up, rt_down = _rows_per_tg(F, FP4_RT_UP), _rows_per_tg(D, FP4_RT_DOWN)
        tl = K["map"](
            inputs=[indices],
            template=[("TK", n), ("G", FP4_G)],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256), 1, 1),
            output_shapes=[(n * (FP4_G + 1),)],
            output_dtypes=[mx.uint32],
        )[0]
        h = K["group_gate_up"](
            inputs=[x, *w, indices, tl],
            template=base + [("G", FP4_G), ("NR", F), ("NK", D), ("RT", rt_up)],
            grid=(256 * F // rt_up, n, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, F)],
            output_dtypes=[x.dtype],
        )[0]
        y = K["group_down"](
            inputs=[h, dq, ds, bd, indices, ones, tl],
            template=base + [("G", FP4_G), ("NR", D), ("NK", F), ("RT", rt_down)],
            grid=(256 * D // rt_down, n, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, D)],
            output_dtypes=[mx.float32],
        )[0]
    return y.astype(x.dtype)


def _reference_fp4(x, indices, weights, act):
    gq, gs, uq, us, dq, ds, bg, bu, bd = weights
    xx = mx.expand_dims(x, (-2, -3))
    mm = partial(mx.gather_qmm, rhs_indices=indices, transpose=True, mode="mxfp4")
    g = mm(xx, gq, gs) + mx.expand_dims(bg[indices], -2)
    u = mm(xx, uq, us) + mx.expand_dims(bu[indices], -2)
    if act == 0:
        a = nn.silu(g) * u
    else:
        g, u = mx.minimum(g, 7.0), mx.clip(u, -7.0, 7.0)
        a = g * mx.sigmoid(1.702 * g) * (u + 1.0)
    y = mm(a.astype(x.dtype), dq, ds) + mx.expand_dims(bd[indices], -2)
    return y.squeeze(-2).reshape(-1, x.shape[-1])


def _make_fp4_op(act):
    @mx.custom_function
    def op(x, indices, *weights):
        return _forward_fp4(x, indices, weights, act)

    @op.vjp
    def op(primals, cotangent, output):
        x, indices, *weights = primals
        _, (dx,) = mx.vjp(
            lambda x: _reference_fp4(x, indices, weights, act), [x], [cotangent]
        )
        return (dx, mx.zeros_like(indices), *(mx.zeros_like(w) for w in weights))

    return op


_fp4_ops = {0: _make_fp4_op(0), 1: _make_fp4_op(1)}

FP4_MAX_LOAD = 16  # average tokens per expert up to which these kernels beat gather_qmm (gpt-oss shapes)


def _switch_glu_fp4(glu, x, indices):
    projs = (glu.gate_proj, glu.up_proj, glu.down_proj)
    act = getattr(glu.activation, "metal_act_fp4", None)
    if _fp4_kernels is None or act is None or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    for p in projs:
        fmt = (
            getattr(p, "bits", None),
            getattr(p, "group_size", None),
            getattr(p, "mode", None),
        )
        if fmt != (4, 32, "mxfp4") or "bias" not in p:
            return None
    k = indices.shape[-1]
    T = indices.size // k
    E, D = glu.gate_proj.weight.shape[0], x.shape[-1]
    if T * k / E > FP4_MAX_LOAD:
        return None
    weights = [a for p in projs for a in (p.weight, p.scales)]
    weights += [p.bias.astype(mx.float32) for p in projs]
    y = _fp4_ops[act](
        x.reshape(T, D), indices.reshape(T, k).astype(mx.uint32), *weights
    )
    return y.reshape(*indices.shape, D)
