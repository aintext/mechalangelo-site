# Assemblaggio e rendering di una cornice 9-slice a partire da uno "stile".
import json
import math
import os
import numpy as np
from scipy import ndimage as ndi
from motore import (SS, NEG, f32, pnoise, build_profile, Carve, cast_shadow, down, save_webp)

L = np.array([-0.22, -0.78, 0.62], f32)  # luce da museo: dall'alto, leggermente da sinistra
L /= np.linalg.norm(L)
V = np.array([0, 0, 1], f32)
HV = (L + V) / np.linalg.norm(L + V)

# materiali: 0 oro, 1 nero ebanizzato, 2 lino, 3 oro della battuta (piu' chiaro)
GOLDS = {
    "chiaro": dict(alb=(0.80, 0.60, 0.30), spec=(1.00, 0.84, 0.52), pat=(0.20, 0.14, 0.07), bole=(0.55, 0.26, 0.13)),
    "rosato": dict(alb=(0.82, 0.60, 0.40), spec=(1.00, 0.83, 0.64), pat=(0.22, 0.13, 0.08), bole=(0.60, 0.28, 0.18)),
    "brunito": dict(alb=(0.62, 0.44, 0.22), spec=(0.92, 0.72, 0.42), pat=(0.13, 0.09, 0.05), bole=(0.42, 0.20, 0.10)),
    "bronzo": dict(alb=(0.48, 0.34, 0.18), spec=(0.82, 0.62, 0.36), pat=(0.10, 0.075, 0.045), bole=(0.30, 0.17, 0.09)),
    "antico": dict(alb=(0.72, 0.52, 0.27), spec=(0.98, 0.79, 0.48), pat=(0.17, 0.11, 0.06), bole=(0.55, 0.22, 0.12)),
}


def nmap(H):
    gy, gx = np.gradient(H)
    nx, ny = -gx, -gy
    l = np.sqrt(nx * nx + ny * ny + 1)
    return nx / l, ny / l, 1 / l


def _put(H, orn, sl, hc, sink):
    """L'ornamento prevale sul profilo nella sua impronta (il profilo sporge solo se molto piu' alto)."""
    sub = H[sl]
    w = hc > NEG / 2
    nv = np.where(w, np.maximum(hc, np.where(sub > NEG / 2, sub - sink, NEG)), sub)
    o = w & (hc >= nv - 1e-3)
    sub[:] = nv
    orn[sl] |= o


def render(st, with_center, outdir=None):
    T, S, M = st["T"] * SS, st["S"] * SS, st["M"] * SS
    N = 2 * S + M
    m0 = st.get("margin", 0.0) * T
    yy, xx = np.mgrid[0:N, 0:N].astype(f32) + 0.5
    d = np.stack([yy, N - xx, N - yy, xx])  # alto, destra, basso, sinistra
    side = d.argmin(0)
    v = d.min(0)
    u = np.where((side == 0) | (side == 2), xx - S, yy - S)
    del d
    # profilo
    prof, pmat, rng = build_profile(int(T - m0), st["profile"])
    vp = v - m0
    band = (vp >= 0) & (v < T)
    depth = st.get("depth", 1.0) * T
    H = np.where(band, np.interp(vp, np.arange(len(prof)) + 0.5, prof) * depth, NEG).astype(f32)
    mat = np.where(band, pmat[np.clip(vp.astype(int), 0, len(pmat) - 1)], 0).astype(np.int8)
    # rumore periodico lungo il lato (periodo M): grana del gesso e usura
    Tn = int(S)
    nz_f = pnoise((Tn, M), 1.5, st.get("seed", 1))
    nz_w = pnoise((Tn, M), 1.7, st.get("seed", 1) + 7)
    nz_g = pnoise((Tn, M), 1.0, st.get("seed", 1) + 3, aniso=3.0)  # venatura lungo il lato
    ui = np.mod(np.floor(u).astype(np.int64), M)
    vi = np.clip(v.astype(np.int64), 0, Tn - 1)
    NF, NW, NG = nz_f[vi, ui], nz_w[vi, ui], nz_g[vi, ui]
    # motivi continui (perline, foglie, onde...), periodici in u
    if "motif" in st:
        U = np.mod(u, M)
        add, mm = st["motif"](U, vp, rng, T, depth)
        H = np.where(band, H + add, H)
        if mm is not None:
            mat = np.where(band & (mm >= 0), mm, mat).astype(np.int8)
    # ornamenti d'angolo
    orn = np.zeros((N, N), bool)
    sink = st.get("sink", 0.3) * depth
    if "corner" in st:
        c = Carve(0, 0, S, S)
        st["corner"](c, T, S, depth)
        Hc = np.maximum(c.H, c.H.T)
        for sl, fl in (((slice(0, S), slice(0, S)), (1, 1)), ((slice(0, S), slice(N - S, N)), (1, -1)),
                       ((slice(N - S, N), slice(0, S)), (-1, 1)), ((slice(N - S, N), slice(N - S, N)), (-1, -1))):
            hc = Hc[::fl[0], ::fl[1]]
            _put(H, orn, sl, hc, sink)
    if with_center and "center" in st:
        Lc = st["Lc"] * SS
        c = Carve(-Lc / 2, 0, Lc, S)
        st["center"](c, T, S, depth)
        Hc = np.maximum(c.H, c.H[:, ::-1])
        a0 = N // 2 - Lc // 2
        regs = [((slice(0, S), slice(a0, a0 + Lc)), Hc), ((slice(N - S, N), slice(a0, a0 + Lc)), Hc[::-1, :]),
                ((slice(a0, a0 + Lc), slice(0, S)), Hc.T), ((slice(a0, a0 + Lc), slice(N - S, N)), Hc.T[:, ::-1])]
        for sl, hc in regs:
            _put(H, orn, sl, hc, sink)
    mat = np.where(orn, st.get("orn_mat", 0), mat).astype(np.int8)
    solid = H > NEG / 2
    # grana fine del gesso e piccole irregolarita' dell'intaglio
    H = np.where(solid, H + NF * st.get("rough", 0.0012) * T + NG * 0.0008 * T, H)
    pic = (~solid) & (v >= T * 0.98)
    wall = (~solid) & ~pic
    Hf = np.where(solid, H, np.where(pic, -0.04 * T, -0.02 * T)).astype(f32)
    # ombre e occlusione
    occ = cast_shadow(Hf, L, -0.02 * T)
    lit = 1 - np.clip(occ / (0.012 * T), 0, 1) * 0.86
    b1 = ndi.gaussian_filter(Hf, 0.012 * T)
    b2 = ndi.gaussian_filter(Hf, 0.045 * T)
    cav = np.clip((b1 - Hf) / (0.010 * T), 0, 1) * 0.55 + np.clip((b2 - Hf) / (0.03 * T), 0, 1) * 0.45
    conv = np.clip((Hf - b1) / (0.006 * T), 0, 1)
    nx, ny, nzz = nmap(Hf)
    ndl = np.clip(nx * L[0] + ny * L[1] + nzz * L[2], 0, 1)
    ndh = np.clip(nx * HV[0] + ny * HV[1] + nzz * HV[2], 0, 1)
    env = np.clip(0.45 - 0.75 * ny + 0.1 * nx, 0, 1)  # soffitto chiaro sopra, buio sotto
    rim = np.clip(-ny, 0, 1)
    g = GOLDS[st.get("gold", "chiaro")]
    alb, spc, pat, bole = (np.array(g[k], f32) for k in ("alb", "spec", "pat", "bole"))
    gloss = st.get("gloss", 1.0)
    ex = lambda a: a[..., None]
    # metallo: riflette l'ambiente (soffitto chiaro, pavimento scuro) + riflesso della luce
    rx, ry, rz = 2 * nzz * nx, 2 * nzz * ny, 2 * nzz * nzz - 1
    up = -ry
    envm = 0.30 + 0.95 * np.clip((up + 0.15) / 0.8, 0, 1) ** 1.3 - 0.22 * np.clip(-up / 0.6, 0, 1) + 0.12 * rx * 0
    envm = envm * (1 - 0.55 * np.clip(cav, 0, 1))
    rl = np.clip(rx * L[0] + ry * L[1] + rz * L[2], 0, 1)
    spec = (1.3 * rl ** 60 + 0.45 * rl ** 9) * lit * gloss
    C = spc * ex(envm * (0.45 + 0.55 * lit) * 0.78 + spec) + alb * ex(0.10 + 0.22 * ndl * lit)
    # usura: sui rilievi affiora il bolo; nelle cavita' patina scura e sporco
    wear = np.clip((conv * 1.1 + NW * 0.5 - 0.95) * 2.2, 0, 1) * st.get("wear", 0.5)
    C = C * ex(1 - wear) + bole * ex(wear * (0.30 + 0.7 * ndl * lit))
    dirt = np.clip(cav * 1.3 + NW * 0.10, 0, 0.94) * st.get("patina", 0.8)
    C = C * ex(1 - dirt) + pat * ex(dirt)
    C *= ex(1 - 0.22 * np.clip(-NW * 0.6, 0, 1) * st.get("mottle", 0.6))
    # nero ebanizzato
    if (mat == 1).any():
        k = mat == 1
        Cb = np.array([0.045, 0.038, 0.034], f32) * ex(0.4 + 0.9 * ndl * lit) \
            + np.array([0.55, 0.52, 0.48], f32) * ex((0.55 * ndh ** 50 + 0.10 * ndh ** 6) * lit + 0.06 * env ** 2)
        Cb *= ex(1 - 0.5 * np.clip(cav, 0, 1)) * ex(1 + 0.08 * NG)
        C[k] = Cb[k]
    # lino chiaro (ottocentesca)
    if (mat == 2).any():
        k = mat == 2
        weave = (np.sin(xx * 2 * math.pi / 2.6) * np.sin(yy * 2 * math.pi / 2.6)) * 0.035
        Cl = np.array([0.86, 0.82, 0.72], f32) * ex((0.45 + 0.55 * ndl * lit) * (1 + weave + 0.03 * NF)) * ex(1 - 0.4 * cav)
        C[k] = Cl[k]
    C *= st.get("exposure", 1.0)
    # alfa: cornice + ombra portata sul dipinto
    a = solid.astype(f32)
    vv = np.clip((v - T) / (0.05 * T), 0, 50)
    sh_pic = ndi.gaussian_filter(np.clip(occ / (0.03 * T), 0, 1), 0.02 * T) * 0.5 + 0.30 * np.exp(-vv)
    a = np.where(pic, np.clip(sh_pic, 0, 0.8), a)
    C[pic] = np.array([0.08, 0.05, 0.03], f32)
    # ombra di contatto sul muro dietro le sporgenze (solo un velo)
    C, a = down(C, a)
    info = dict(N=N // SS, S=S // SS, T=T // SS, M=M // SS, margin=round(m0 / T, 4))
    return C, a, info, orn


def write(st, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    C, a, info, _ = render(st, False)
    save_webp(os.path.join(outdir, f"{name}.webp"), C, a)
    out = dict(info)
    if "center" in st:
        C2, a2, _, _ = render(st, True)
        diff = np.abs(C2 - C).sum(-1) * np.maximum(a, a2) + np.abs(a2 - a)
        msk = ndi.binary_dilation(diff > 0.02, iterations=3)
        msk = ndi.gaussian_filter(msk.astype(f32), 2.0)
        N, S, Lc = info["N"], info["S"], st["Lc"]
        a0 = N // 2 - Lc // 2
        al = np.clip(msk * 1.15, 0, 1) * a2
        crops = {"t": (slice(0, S), slice(a0, a0 + Lc)), "b": (slice(N - S, N), slice(a0, a0 + Lc)),
                 "l": (slice(a0, a0 + Lc), slice(0, S)), "r": (slice(a0, a0 + Lc), slice(N - S, N))}
        for k, sl in crops.items():
            save_webp(os.path.join(outdir, f"{name}-{k}.webp"), C2[sl], al[sl])
        out["Lc"] = Lc
        # anteprima completa
        prev = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anteprime")
        os.makedirs(prev, exist_ok=True)
        save_webp(os.path.join(prev, f"_{name}-full.webp"), C2, a2)
    return out
