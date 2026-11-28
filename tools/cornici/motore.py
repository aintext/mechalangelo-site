# Motore procedurale per le cornici antiche del sito (9-slice).
# Profilo -> mappa di altezza con intagli -> normali -> doratura con patina, usura e ombre portate.
# Tutto in numpy: nessun servizio esterno. Uso: python3 tools/cornici/genera.py
import math
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

f32 = np.float32
SS = 2  # sovracampionamento interno
NEG = -1e9


def pnoise(shape, beta, seed, aniso=1.0):
    """Rumore frattale periodico (FFT): periodico su entrambi gli assi."""
    rng = np.random.default_rng(seed)
    h, w = shape
    ky = np.fft.fftfreq(h)[:, None] * aniso
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1
    spec = (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / k ** beta
    spec[0, 0] = 0
    n = np.fft.irfft2(spec, s=shape).astype(f32)
    return (n - n.mean()) / (n.std() + 1e-9)


# ---------------------------------------------------------------- profilo
def build_profile(T, segs):
    """segs: dict con w (frazione), k (tipo), h0, h1, b (rigonfiamento), mat, name.
    Restituisce altezza(v) e materiale(v) campionati per pixel interno, e gli intervalli per nome."""
    tot = sum(s["w"] for s in segs)
    v = np.arange(T, dtype=f32) + 0.5
    h = np.zeros(T, f32)
    mat = np.zeros(T, np.int8)
    rng = {}
    pos = 0.0
    for s in segs:
        a = pos * T / tot
        b = (pos + s["w"]) * T / tot
        pos += s["w"]
        m = (v >= a) & (v < b)
        t = (v[m] - a) / max(b - a, 1e-6)
        k = s.get("k", "lin")
        h0, h1 = s.get("h0", 0), s.get("h1", s.get("h0", 0))
        if k == "lin":
            f = t
        elif k == "qr":  # quarto di tondo convesso
            f = np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))
        elif k == "cove":  # gola concava
            f = 1 - np.sqrt(np.clip(1 - t ** 2, 0, 1))
        elif k == "ogee":
            f = 0.5 - 0.5 * np.cos(np.pi * t)
        else:
            f = t
        y = h0 + (h1 - h0) * f
        bb = s.get("b", 0)
        if bb:
            y = y + bb * np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1))
        h[m] = y
        mat[m] = s.get("mat", 0)
        if "name" in s:
            rng[s["name"]] = (a, b)
    return h, mat, rng


# ---------------------------------------------------------------- intagli (coordinate cartesiane locali)
class Carve:
    """Area di intaglio locale: H (altezza assoluta, NEG = vuoto)."""

    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, int(w), int(h)
        self.H = np.full((self.h, self.w), NEG, f32)

    def _box(self, px, py, r):
        j0 = max(0, int(px - r - self.x0)); j1 = min(self.w, int(px + r - self.x0) + 2)
        i0 = max(0, int(py - r - self.y0)); i1 = min(self.h, int(py + r - self.y0) + 2)
        return i0, i1, j0, j1

    def chain(self, pts, rads, hs=0.6, base=0.0):
        """Tubo a sezione tonda (schiacciata di hs) lungo una polilinea campionata."""
        for (px, py), r in zip(pts, rads):
            if r <= 0.6:
                continue
            i0, i1, j0, j1 = self._box(px, py, r)
            if i0 >= i1 or j0 >= j1:
                continue
            xs = np.arange(j0, j1, dtype=f32) + self.x0 + 0.5 - px
            ys = np.arange(i0, i1, dtype=f32) + self.y0 + 0.5 - py
            d2 = xs[None, :] ** 2 + ys[:, None] ** 2
            hh = base + hs * np.sqrt(np.maximum(r * r - d2, 0))
            hh[d2 > r * r] = NEG
            sub = self.H[i0:i1, j0:j1]
            np.maximum(sub, hh, out=sub)

    def groove(self, pts, width, depth):
        """Solco (scanalatura) lungo una curva: toglie altezza dove c'e' gia' materia."""
        D = {}
        r = width * 3
        for px, py in pts:
            i0, i1, j0, j1 = self._box(px, py, r)
            if i0 >= i1 or j0 >= j1:
                continue
            xs = np.arange(j0, j1, dtype=f32) + self.x0 + 0.5 - px
            ys = np.arange(i0, i1, dtype=f32) + self.y0 + 0.5 - py
            d2 = xs[None, :] ** 2 + ys[:, None] ** 2
            key = (i0, i1, j0, j1)
            g = np.exp(-d2 / (width * width)).astype(f32)
            sub = self.H[i0:i1, j0:j1]
            # profondita' massima, non cumulativa: si usa una maschera locale
            if not hasattr(self, "_G"):
                self._G = np.zeros_like(self.H)
            gs = self._G[i0:i1, j0:j1]
            np.maximum(gs, g, out=gs)
        if hasattr(self, "_G"):
            m = self.H > NEG / 2
            self.H[m] -= depth * self._G[m]
            del self._G

    def field(self, fn, bbox=None):
        """Aggiunge un campo analitico fn(X,Y)->altezza (NEG fuori)."""
        if bbox is None:
            i0, i1, j0, j1 = 0, self.h, 0, self.w
        else:
            x0, y0, x1, y1 = bbox
            j0 = max(0, int(x0 - self.x0)); j1 = min(self.w, int(x1 - self.x0) + 1)
            i0 = max(0, int(y0 - self.y0)); i1 = min(self.h, int(y1 - self.y0) + 1)
            if i0 >= i1 or j0 >= j1:
                return
        X = np.arange(j0, j1, dtype=f32)[None, :] + self.x0 + 0.5
        Y = np.arange(i0, i1, dtype=f32)[:, None] + self.y0 + 0.5
        X, Y = np.broadcast_arrays(X, Y)
        hh = fn(X, Y).astype(f32)
        sub = self.H[i0:i1, j0:j1]
        np.maximum(sub, hh, out=sub)


def sample_curve(fn, n):
    t = np.linspace(0, 1, n, dtype=f32)
    return fn(t), t


def leaf(c, p0, ang, L, W, curl=0.0, lobes=3, hs=0.55, base=0.0, lobe_side=0, rib=True, tipcurl=0.0):
    """Foglia d'acanto: asse curvo, larghezza a goccia, lobi laterali arricciati, nervatura centrale."""
    n = max(12, int(L / max(W * 0.12, 1.0)))
    t = np.linspace(0, 1, n, dtype=f32)
    th = ang + curl * t ** 1.4 + tipcurl * np.clip((t - 0.75) / 0.25, 0, 1) ** 2
    step = L / n
    xs = p0[0] + np.cumsum(np.cos(th)) * step
    ys = p0[1] + np.cumsum(np.sin(th)) * step
    w = W * np.clip(np.sin(np.pi * np.clip(t, 0, 1) ** 0.72), 0, 1) ** 0.75 * 0.5 + 0.6
    w[0] = max(w[0], W * 0.18)
    pts = np.stack([xs, ys], 1)
    c.chain(pts, w, hs, base)
    for k in range(lobes):
        tk = 0.22 + 0.62 * (k + 0.5) / lobes
        i = min(n - 1, int(tk * n))
        sides = [1, -1] if lobe_side == 0 else [lobe_side]
        for s in sides:
            a = th[i] + s * (0.95 - 0.3 * tk)
            px = xs[i] + math.cos(th[i] + s * math.pi / 2) * w[i] * 0.45
            py = ys[i] + math.sin(th[i] + s * math.pi / 2) * w[i] * 0.45
            leaf(c, (px, py), a, L * (0.42 - 0.18 * tk), W * (0.5 - 0.15 * tk), curl=-s * 1.4,
                 lobes=0, hs=hs, base=base, rib=False)
    if rib:
        c.groove(pts[2:-2], max(1.2, W * 0.05), hs * W * 0.12)
    return pts


def scroll(c, cen, r0, r1, a0, a1, t0, t1, hs=0.6, base=0.0, flute=True, knob=True):
    """Voluta a C: spirale da raggio r0 a r1 tra gli angoli a0 e a1."""
    n = int(abs(a1 - a0) * max(r0, r1) / max(min(t0, t1) * 0.3, 1.0)) + 12
    t = np.linspace(0, 1, n, dtype=f32)
    a = a0 + (a1 - a0) * t
    r = r0 + (r1 - r0) * t
    pts = np.stack([cen[0] + r * np.cos(a), cen[1] + r * np.sin(a)], 1)
    rad = t0 + (t1 - t0) * t
    c.chain(pts, rad, hs, base)
    if knob:
        c.chain([pts[-1]], [t1 * 1.45], hs, base)
    if flute:
        c.groove(pts[3:-3], max(1.0, min(t0, t1) * 0.28), hs * min(t0, t1) * 0.35)
    return pts


def path(c, pts, t0, t1, hs=0.6, base=0.0, flute=False):
    """Tubo lungo una polilinea fitta con spessore da t0 a t1."""
    pts = np.asarray(pts, f32)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    step = max(min(t0, t1) * 0.3, 1.0)
    s = np.arange(0, d[-1] + 1e-3, step)
    x = np.interp(s, d, pts[:, 0]); y = np.interp(s, d, pts[:, 1])
    t = s / max(d[-1], 1e-6)
    P = np.stack([x, y], 1)
    c.chain(P, t0 + (t1 - t0) * t, hs, base)
    if flute:
        c.groove(P[3:-3], max(1.0, min(t0, t1) * 0.28), hs * min(t0, t1) * 0.35)
    return P


def bezier(p0, p1, p2, p3, n=80):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(lambda p: np.asarray(p, float), (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3).astype(f32)


def shell(c, hinge, ang, R, spread, ribs, hs=0.5, base=0.0, scallop=0.07):
    """Conchiglia rocaille: ventaglio di costolature dal cardine."""
    hx, hy = hinge

    def fn(X, Y):
        dx, dy = X - hx, Y - hy
        rr = np.hypot(dx, dy)
        th = np.arctan2(dy, dx) - ang
        th = np.arctan2(np.sin(th), np.cos(th))
        tt = th / spread
        edge = R * (1 + scallop * np.cos(tt * ribs * np.pi)) * (1 - 0.18 * tt ** 2)
        ins = (np.abs(tt) < 1) & (rr < edge)
        q = np.clip(rr / edge, 0, 1)
        dome = np.sqrt(np.clip(1 - q ** 2, 0, 1)) * (0.55 + 0.45 * np.sqrt(np.clip(q * 3, 0, 1)))
        ribp = 0.5 + 0.5 * np.cos(tt * ribs * np.pi)
        h = base + hs * R * (0.55 * dome + 0.22 * dome * ribp ** 1.5) * (1 - 0.35 * tt ** 2)
        return np.where(ins, h, NEG)

    c.field(fn, (hx - R * 1.2, hy - R * 1.2, hx + R * 1.2, hy + R * 1.2))


def rosette(c, cen, R, petals, hs=0.5, base=0.0, rot=0.0):
    cx, cy = cen

    def fn(X, Y):
        dx, dy = X - cx, Y - cy
        rr = np.hypot(dx, dy)
        th = np.arctan2(dy, dx) - rot
        lob = 0.62 + 0.38 * np.abs(np.cos(th * petals / 2)) ** 0.7
        edge = R * lob
        q = np.clip(rr / edge, 0, 1)
        ins = rr < edge
        h = base + hs * R * (0.5 * np.sqrt(np.clip(1 - q ** 2, 0, 1)) * (0.7 + 0.3 * q))
        bud = rr < R * 0.3
        h = np.where(bud, base + hs * R * (0.42 + 0.35 * np.sqrt(np.clip(1 - (rr / (R * 0.3)) ** 2, 0, 1))), h)
        return np.where(ins, h, NEG)

    c.field(fn, (cx - R, cy - R, cx + R, cy + R))


# ---------------------------------------------------------------- utilita' per il rendering
def shift(A, dx, dy, fill):
    """B[y,x] = A[y+dy, x+dx]."""
    B = np.full_like(A, fill)
    h, w = A.shape
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys0 < ys1 and xs0 < xs1:
        B[ys0:ys1, xs0:xs1] = A[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return B


def cast_shadow(H, L, fill):
    """Ombra portata: distanza verticale sotto l'orizzonte verso la luce (a meta' risoluzione)."""
    Hs = H.reshape(H.shape[0] // 2, 2, H.shape[1] // 2, 2).mean((1, 3)) / 2
    dxy = math.hypot(L[0], L[1])
    dx, dy = L[0] / dxy, L[1] / dxy
    slope = L[2] / dxy
    occ = np.zeros_like(Hs)
    maxh = float(Hs.max() - Hs.min())
    steps = int(maxh / slope) + 3
    for k in range(1, steps):
        sx, sy = int(round(dx * k)), int(round(dy * k))
        S = shift(Hs, sx, sy, fill / 2)
        np.maximum(occ, S - k * slope - Hs, out=occ)
    occ = ndi.zoom(occ, 2, order=1)[:H.shape[0], :H.shape[1]] * 2
    return occ


def down(rgb, a):
    """Riduzione 2x con alfa premoltiplicata."""
    h, w = a.shape
    pm = rgb * a[..., None]
    pm = pm.reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))
    aa = a.reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    out = pm / np.maximum(aa, 1e-6)[..., None]
    return out, aa


def save_webp(path, rgb, a, q=90):
    img = np.dstack([np.clip(rgb, 0, 1) * 255, np.clip(a, 0, 1) * 255]).round().astype(np.uint8)
    im = Image.fromarray(img, "RGBA")
    im.save(path, "WEBP", quality=q, alpha_quality=100, method=6)
    return im


# ---------------------------------------------------------------- foglia d'acanto da silhouette
from PIL import ImageDraw


def _axis(p0, ang, L, curl, tipcurl, n):
    t = np.linspace(0, 1, n)
    th = ang + curl * t ** 1.3 + tipcurl * np.clip((t - 0.7) / 0.3, 0, 1) ** 2
    step = L / (n - 1)
    xs = p0[0] + np.r_[0, np.cumsum(np.cos(th[:-1]))] * step
    ys = p0[1] + np.r_[0, np.cumsum(np.sin(th[:-1]))] * step
    return t, th, xs, ys


def acanthus(c, p0, ang, L, W, curl=0.0, tipcurl=0.0, lobes=4, hs=0.5, base=0.0, serr=6, rib=0.14, side=0,
             fat=1.0):
    """Foglia d'acanto: silhouette lobata e dentellata, rilievo arrotondato (distanza dal bordo),
    nervatura centrale e venature verso le punte dei lobi."""
    n = 160
    t, th, xs, ys = _axis(p0, ang, L, curl, tipcurl, n)
    env = np.clip(np.sin(np.pi * np.clip(t, 0, 1) ** 0.62), 0, 1) ** 0.8 * (1 - 0.25 * t)
    env = np.maximum(env, 0.18 * (t < 0.08))
    k = np.clip(t * lobes, 0, lobes)
    lobe = np.abs(np.sin(np.pi * k)) ** 0.55  # lobi tondeggianti
    tooth = 1 - 0.18 * np.abs(np.sin(np.pi * k * serr / 2)) ** 3  # dentelli
    wl = W * 0.5 * env * (0.50 + 0.50 * lobe) * tooth
    wr = wl.copy()
    if side == 1:
        wr *= 0.45
    elif side == -1:
        wl *= 0.45
    nxv, nyv = -np.sin(th), np.cos(th)
    left = np.stack([xs + nxv * wl, ys + nyv * wl], 1)
    right = np.stack([xs - nxv * wr, ys - nyv * wr], 1)
    poly = np.concatenate([left, right[::-1]])
    pad = 4
    x0, y0 = poly.min(0) - pad
    x1, y1 = poly.max(0) + pad
    w, h = int(x1 - x0) + 1, int(y1 - y0) + 1
    if w < 3 or h < 3:
        return
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in (poly - [x0, y0])], fill=255)
    msk = np.asarray(im) > 127
    d = ndi.distance_transform_edt(msk)
    ax = Image.new("L", (w, h), 255)
    dr = ImageDraw.Draw(ax)
    axp = np.stack([xs, ys], 1) - [x0, y0]
    dr.line([tuple(p) for p in axp[3:-6]], fill=0, width=1)
    # venature: dall'asse verso le punte dei lobi
    vn = Image.new("L", (w, h), 255)
    dv = ImageDraw.Draw(vn)
    for j in range(lobes):
        tk = (j + 0.5) / lobes
        i = int(tk * (n - 1))
        i2 = min(n - 1, i + 6)
        for sgn, ww in ((1, wl), (-1, wr)):
            ex_ = xs[i2] + sgn * nxv[i2] * ww[i2] * 0.85 - x0
            ey_ = ys[i2] + sgn * nyv[i2] * ww[i2] * 0.85 - y0
            dv.line([(xs[max(0, i - 4)] - x0, ys[max(0, i - 4)] - y0), (ex_, ey_)], fill=0, width=1)
    da = ndi.distance_transform_edt(np.asarray(ax) > 0)
    dvv = ndi.distance_transform_edt(np.asarray(vn) > 0)
    q = np.clip(d / (0.30 * W * fat), 0, 1)
    dome = np.sqrt(1 - (1 - q) ** 2)
    hgt = base + hs * W * 0.5 * (dome * 0.85 + 0.15 * q)
    hgt -= hs * W * 0.5 * rib * np.exp(-(da / max(1.5, W * 0.045)) ** 2)
    hgt -= hs * W * 0.5 * rib * 0.5 * np.exp(-(dvv / max(1.2, W * 0.03)) ** 2) * (q > 0.25)
    hgt = np.where(msk, hgt, NEG).astype(f32)
    # incolla nel Carve
    gx0, gy0 = int(math.floor(x0 - c.x0)), int(math.floor(y0 - c.y0))
    i0, j0 = max(0, gy0), max(0, gx0)
    i1, j1 = min(c.h, gy0 + h), min(c.w, gx0 + w)
    if i0 >= i1 or j0 >= j1:
        return
    sub = c.H[i0:i1, j0:j1]
    np.maximum(sub, hgt[i0 - gy0:i1 - gy0, j0 - gx0:j1 - gx0], out=sub)


def cartouche(c, cen, rx, ry, tube, hs=0.6, base=0.0, dish=0.25):
    """Cartiglio ovale: cornice a tubo con fondo concavo."""
    cx, cy = cen
    a = np.linspace(0, 2 * np.pi, 220)
    pts = np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1).astype(f32)
    c.chain(pts, np.full(len(pts), tube, f32), hs, base)

    def fn(X, Y):
        q = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        return np.where(q < 1, base + hs * tube * 0.2 + dish * tube * q, NEG)

    c.field(fn, (cx - rx, cy - ry, cx + rx, cy + ry))
