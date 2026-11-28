# Motivi continui lungo il profilo, in coordinate (U lungo il lato, periodico; v dal bordo esterno).
import math
import numpy as np

f32 = np.float32


def seg(vp, r):
    a, b = r
    return (vp - a) / max(b - a, 1e-6), (vp >= a) & (vp < b), (b - a)


def beads(U, vp, r, p, k, elong=0.85, gap=0.08):
    """Perline (fila di perle) sul segmento r, periodo p px."""
    t, m, wd = seg(vp, r)
    s = (np.mod(U, p) - p / 2) / (p * (0.5 - gap / 2) * elong)
    q = (t - 0.5) * 2
    d2 = s * s + q * q
    h = k * wd * 0.5 * np.sqrt(np.clip(1 - d2, 0, 1))
    return np.where(m, h, 0)


def pearls_and_reels(U, vp, r, p, k):
    """Perline alternate a fusarole (una perla tonda e due dischetti)."""
    t, m, wd = seg(vp, r)
    c = np.mod(U, p) / p
    q = (t - 0.5) * 2
    s1 = (c - 0.33) / 0.27
    pearl = np.sqrt(np.clip(1 - s1 ** 2 - q ** 2, 0, 1))
    s2 = (np.mod(c - 0.75, 1) - 0.5 + 0.5)
    reel = np.zeros_like(c)
    for cc in (0.72, 0.86):
        s3 = (c - cc) / 0.05
        reel = np.maximum(reel, 0.55 * np.sqrt(np.clip(1 - s3 ** 2 - (q / 0.8) ** 2, 0, 1)))
    return np.where(m, k * wd * 0.5 * np.maximum(pearl, reel), 0)


def leaf_tips(U, vp, r, p, k, inward=True, serr=0.0):
    """Foglie a punta (lambrecchini) che puntano verso l'interno (o l'esterno)."""
    t, m, wd = seg(vp, r)
    if not inward:
        t = 1 - t
    s = (np.mod(U, p) - p / 2) / (p / 2)
    w = 0.95 * np.clip(1 - t, 0, 1) ** 0.85 * np.sqrt(np.clip(t * 5, 0, 1))
    w = w * (1 + serr * np.abs(np.sin(t * math.pi * 4)))
    q = np.abs(s) / np.maximum(w, 1e-4)
    leafh = np.sqrt(np.clip(1 - q * q, 0, 1)) * (0.6 + 0.4 * t)
    leafh -= 0.35 * np.exp(-(s / 0.07) ** 2) * (q < 1) * np.clip(1 - t, 0, 1)
    # piccolo dardo fra le foglie
    s2 = (np.mod(U + p / 2, p) - p / 2) / (p / 2)
    dart = np.sqrt(np.clip(1 - (s2 / (0.18 * np.clip(1.2 - t, 0, 1) + 1e-3)) ** 2, 0, 1)) * 0.5 * (t < 0.85)
    return np.where(m, k * wd * np.maximum(leafh, dart), 0)


def acanthus_run(U, vp, r, p, k, tilt=0.55, lobes=3):
    """Fascia continua di foglie d'acanto inclinate e sovrapposte."""
    t, m, wd = seg(vp, r)
    out = np.zeros_like(U)
    for off, sc in ((0.0, 1.0), (0.5, 0.82)):
        sh = U + (t - 0.5) * wd * tilt * 2 + off * p
        s = (np.mod(sh, p) - p / 2) / (p / 2) / 0.95
        tt = np.clip(t, 0, 1)
        w = np.clip(np.sin(math.pi * tt ** 0.8), 0, 1) ** 0.7
        w = w * (0.78 + 0.22 * np.abs(np.cos(tt * math.pi * lobes)))  # lobi
        q = np.abs(s) / np.maximum(w, 1e-4)
        h = np.sqrt(np.clip(1 - q * q, 0, 1)) * (0.55 + 0.45 * np.sin(math.pi * tt))
        h -= 0.3 * np.exp(-(s / 0.06) ** 2) * (q < 1)  # nervatura
        h -= 0.18 * np.exp(-((np.abs(s) - 0.45 * w) / 0.05) ** 2) * (q < 0.9) * (tt > 0.2)  # venature
        out = np.maximum(out, h * sc)
    return np.where(m, k * wd * 0.5 * out, 0)


def foliage(U, vp, r, p, k):
    """Fogliame continuo fiorentino: foglie lanceolate alternate, sovrapposte, con bacche."""
    t, m, wd = seg(vp, r)
    out = np.zeros_like(U)
    for off, dirn in ((0.0, 1), (0.5, -1)):
        sh = U + dirn * (t - 0.5) * wd * 0.9 + off * p
        c = np.mod(sh, p) / p  # 0..1 lungo la foglia
        q = (t - 0.5) * 2 * dirn  # attraverso
        along = c
        w = 0.82 * np.clip(np.sin(math.pi * np.clip(along, 0, 1) ** 0.75), 0, 1) ** 0.8
        w *= 0.85 + 0.15 * np.abs(np.cos(along * math.pi * 4))
        qq = np.abs(q - 0.25 * np.sin(along * math.pi)) / np.maximum(w, 1e-4)
        h = np.sqrt(np.clip(1 - qq * qq, 0, 1)) * (0.5 + 0.5 * np.sin(math.pi * along))
        h -= 0.28 * np.exp(-((q - 0.25 * np.sin(along * math.pi)) / 0.07) ** 2) * (qq < 1)
        out = np.maximum(out, h)
    # bacche
    c = (np.mod(U + p * 0.25, p * 0.5) - p * 0.25) / (wd * 0.13)
    q = (t - 0.5) / 0.13 * 2
    berry = np.sqrt(np.clip(1 - c * c - q * q, 0, 1)) * 0.75
    out = np.maximum(out, berry)
    return np.where(m, k * wd * 0.5 * out, 0)


def gadroons(U, vp, r, p, k, tilt=0.6):
    """Baccellature oblique."""
    t, m, wd = seg(vp, r)
    sh = U + (t - 0.5) * wd * tilt * 2
    s = (np.mod(sh, p) - p / 2) / (p / 2)
    q = (t - 0.5) * 2
    h = np.sqrt(np.clip(1 - s * s, 0, 1)) * np.sqrt(np.clip(1 - q ** 4, 0, 1))
    return np.where(m, k * wd * 0.5 * h, 0)


def ripple(U, vp, r, p, k, q=None):
    """Modanatura ondulata olandese (a onde)."""
    t, m, wd = seg(vp, r)
    qq = q or wd * 0.9
    h = 0.5 + 0.5 * np.cos(2 * math.pi * U / p + 2.2 * np.sin(2 * math.pi * (vp - r[0]) / qq))
    return np.where(m, k * wd * 0.12 * h, 0)


def egg_dart(U, vp, r, p, k):
    """Ovoli e dardi."""
    t, m, wd = seg(vp, r)
    s = (np.mod(U, p) - p / 2) / (p / 2)
    q = (t - 0.5) * 2
    egg = np.sqrt(np.clip(1 - (s / 0.62) ** 2 - (q / 0.78) ** 2, 0, 1))
    rim = np.clip(1 - np.abs(np.sqrt((s / 0.82) ** 2 + (q / 0.95) ** 2) - 1) / 0.12, 0, 1) * 0.55
    s2 = (np.mod(U + p / 2, p) - p / 2) / (p / 2)
    dart = np.sqrt(np.clip(1 - (s2 / (0.14 * np.clip(1 - t, 0, 1) + 0.01)) ** 2, 0, 1)) * 0.6 * (t > 0.05)
    h = np.maximum(np.maximum(egg, rim * (q < 0.6)), dart)
    return np.where(m, k * wd * 0.5 * h, 0)


def palmettes(U, vp, r, p, k):
    """Palmette in pastiglia (antemio) alternate a piccoli calici."""
    t, m, wd = seg(vp, r)
    c = (np.mod(U, p) - p / 2)
    # base della palmetta verso l'interno (t=1), ventaglio verso l'esterno
    bx, by = c, (1 - t) * wd
    rr = np.hypot(bx, by * 1.0)
    th = np.arctan2(bx, by + 1e-3)
    R = wd * 0.95
    lob = 0.5 + 0.5 * np.cos(th * 7)
    ins = (np.abs(th) < 1.25) & (rr < R * (0.72 + 0.28 * lob))
    h = np.where(ins, np.sqrt(np.clip(1 - (rr / R) ** 2, 0, 1)) * (0.45 + 0.55 * lob), 0)
    core = np.sqrt(np.clip(1 - (bx / (wd * 0.16)) ** 2 - ((by - wd * 0.12) / (wd * 0.2)) ** 2, 0, 1))
    h = np.maximum(h, core * 0.9)
    s2 = (np.mod(U + p / 2, p) - p / 2) / (wd * 0.22)
    cal = np.sqrt(np.clip(1 - s2 ** 2 - ((t - 0.45) / 0.3) ** 2, 0, 1)) * 0.6
    return np.where(m, k * wd * 0.5 * np.maximum(h, cal), 0)


def punch(U, vp, r, p, k):
    """Fondo punzonato (bulinato) a granelli."""
    t, m, wd = seg(vp, r)
    s = np.mod(U, p) - p / 2
    q = np.mod(vp, p) - p / 2
    h = np.sqrt(np.clip(1 - (s * s + q * q) / (p * 0.32) ** 2, 0, 1))
    return np.where(m, -k * p * 0.4 * h, 0)


def flutes(U, vp, r, p, k):
    """Canaletta: scanalatura lungo il lato (gia' nel profilo) con leggere costolature trasversali."""
    t, m, wd = seg(vp, r)
    s = np.mod(U, p) / p
    h = 0.5 + 0.5 * np.cos(2 * math.pi * s)
    return np.where(m, k * wd * 0.05 * h, 0)
