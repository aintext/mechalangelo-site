# Le otto cornici: profili, motivi continui, ornamenti d'angolo e di centro lato.
# Unita': px interni (SS=2). T = spessore fascia, S = slice (>= T, include l'ombra sul dipinto).
import math
import numpy as np
import motivi as mo
from motore import leaf, scroll, shell, rosette, path, bezier, acanthus

PI = math.pi


def P(w, k, h0, h1=None, b=0, mat=0, name=None):
    d = dict(w=w, k=k, h0=h0, h1=h0 if h1 is None else h1, b=b, mat=mat)
    if name:
        d["name"] = name
    return d


# ------------------------------------------------------------ 1. Luigi XV rocaille, oro chiaro
def l15_motif(U, vp, r, T, D):
    add = mo.beads(U, vp, r["perle"], 24, 0.9) + mo.leaf_tips(U, vp, r["punte"], 32, 0.5)
    return add, None


def l15_corner(c, T, S, D):
    b = 0.48 * D
    acanthus(c, (0.86 * T, 0.50 * T), -0.05, 0.86 * T, 0.40 * T, curl=-0.35, tipcurl=-0.9, lobes=4, hs=0.5, base=b)
    acanthus(c, (0.80 * T, 0.78 * T), 0.35, 0.40 * T, 0.22 * T, curl=0.5, tipcurl=0.8, lobes=3, hs=0.45, base=b * 0.95)
    scroll(c, (0.92 * T, 0.26 * T), 0.15 * T, 0.035 * T, PI * 0.85, PI * 2.75, 0.065 * T, 0.03 * T, hs=0.85, base=b)
    path(c, bezier((0.25 * T, 0.10 * T), (0.50 * T, 0.06 * T), (0.75 * T, 0.08 * T), (0.92 * T, 0.11 * T)),
         0.05 * T, 0.035 * T, hs=0.85, base=b * 0.95, flute=True)
    shell(c, (0.60 * T, 0.60 * T), -0.75 * PI, 0.68 * T, 0.95, 11, hs=0.5, base=b)
    rosette(c, (0.62 * T, 0.62 * T), 0.12 * T, 6, hs=0.7, base=b * 1.1)


def l15_center(c, T, S, D):
    b = 0.48 * D
    acanthus(c, (0.30 * T, 0.55 * T), 0.05, 0.80 * T, 0.40 * T, curl=-0.4, tipcurl=-0.9, lobes=4, hs=0.5, base=b)
    scroll(c, (0.36 * T, 0.30 * T), 0.15 * T, 0.035 * T, PI * 1.05, PI * 2.9, 0.065 * T, 0.03 * T, hs=0.85, base=b)
    shell(c, (0, 0.80 * T), -PI / 2, 0.66 * T, 1.0, 13, hs=0.5, base=b)
    rosette(c, (0, 0.80 * T), 0.10 * T, 6, hs=0.7, base=b * 1.1)


LUIGI15 = dict(
    T=190, S=360, M=480, margin=0.06, depth=0.42, gold="chiaro", wear=0.55, patina=0.75, seed=11,
    profile=[P(.07, "qr", 0, .62), P(.07, "lin", .62, .62, b=.12, name="perle"), P(.20, "cove", .62, .22),
             P(.30, "lin", .22, .34, b=.05), P(.13, "ogee", .34, .66), P(.11, "lin", .62, .58, name="punte"),
             P(.12, "lin", .56, .08)],
    motif=l15_motif, corner=l15_corner, center=l15_center, Lc=440,
)

STILI = {"luigi15": LUIGI15}

from motore import cartouche


# ------------------------------------------------------------ 2. Barocca a cartocci, bronzo brunito
def bar_motif(U, vp, r, T, D):
    add = mo.beads(U, vp, r["perle"], 30, 0.9) + mo.acanthus_run(U, vp, r["acanto"], 120, 0.55, tilt=0.6) \
        + mo.beads(U, vp, r["perline"], 24, 0.9)
    return add, None


def bar_corner(c, T, S, D):
    b = 0.55 * D
    acanthus(c, (0.70 * T, 0.30 * T), 0.10, 0.95 * T, 0.48 * T, curl=0.25, tipcurl=1.2, lobes=4, hs=0.55, base=b)
    acanthus(c, (0.55 * T, 0.72 * T), 0.55, 0.55 * T, 0.30 * T, curl=0.4, tipcurl=1.0, lobes=3, hs=0.5, base=b)
    scroll(c, (0.88 * T, 0.66 * T), 0.16 * T, 0.04 * T, -PI * 0.2, PI * 1.7, 0.075 * T, 0.035 * T, hs=0.85, base=b)
    cartouche(c, (0.50 * T, 0.50 * T), 0.30 * T, 0.30 * T, 0.075 * T, hs=0.8, base=b * 1.05)
    shell(c, (0.56 * T, 0.56 * T), -0.75 * PI, 0.36 * T, 0.9, 9, hs=0.45, base=b * 1.05)


def bar_center(c, T, S, D):
    b = 0.55 * D
    acanthus(c, (0.22 * T, 0.40 * T), -0.10, 0.85 * T, 0.46 * T, curl=0.3, tipcurl=1.1, lobes=4, hs=0.55, base=b)
    scroll(c, (0.36 * T, 0.68 * T), 0.14 * T, 0.035 * T, PI * 1.2, PI * 3.0, 0.07 * T, 0.03 * T, hs=0.85, base=b)
    cartouche(c, (0, 0.50 * T), 0.36 * T, 0.30 * T, 0.08 * T, hs=0.8, base=b * 1.05)
    rosette(c, (0, 0.50 * T), 0.16 * T, 8, hs=0.6, base=b * 1.05)


BAROCCA = dict(
    T=230, S=400, M=480, margin=0.05, depth=0.48, gold="bronzo", wear=0.35, patina=0.95, seed=21, exposure=1.08,
    profile=[P(.06, "qr", 0, .55), P(.06, "lin", .55, .55, b=.10, name="perle"), P(.12, "cove", .55, .30),
             P(.34, "lin", .30, .40, b=.30, name="acanto"), P(.05, "lin", .30, .30), P(.12, "cove", .30, .12),
             P(.08, "lin", .18, .18, b=.06, name="perline"), P(.08, "lin", .18, .0)],
    motif=bar_motif, corner=bar_corner, center=bar_center, Lc=480,
)


# ------------------------------------------------------------ 3. Luigi XIV a foglie d'acanto, oro rosato
def l14_motif(U, vp, r, T, D):
    add = mo.egg_dart(U, vp, r["ovoli"], 64, 0.8) + mo.acanthus_run(U, vp, r["acanto"], 96, 0.6, tilt=0.12, lobes=3) \
        + mo.leaf_tips(U, vp, r["punte"], 40, 0.6)
    return add, None


def l14_corner(c, T, S, D):
    b = 0.5 * D
    for ang, L_, W_ in ((-0.75 * PI, 0.62 * T, 0.40 * T), (-0.75 * PI + 0.75, 0.52 * T, 0.32 * T)):
        acanthus(c, (0.62 * T, 0.62 * T), ang, L_, W_, curl=0.2, tipcurl=0.9, lobes=4, hs=0.55, base=b)
    acanthus(c, (0.68 * T, 0.40 * T), -0.05, 0.62 * T, 0.30 * T, curl=-0.2, tipcurl=-1.0, lobes=3, hs=0.5, base=b)
    rosette(c, (0.62 * T, 0.62 * T), 0.14 * T, 8, hs=0.65, base=b * 1.1)


def l14_center(c, T, S, D):
    b = 0.5 * D
    acanthus(c, (0, 0.80 * T), -PI / 2, 0.72 * T, 0.44 * T, curl=0, tipcurl=0, lobes=4, hs=0.55, base=b)
    acanthus(c, (0.04 * T, 0.75 * T), -0.35, 0.70 * T, 0.34 * T, curl=0.2, tipcurl=1.0, lobes=4, hs=0.5, base=b)
    rosette(c, (0, 0.78 * T), 0.12 * T, 8, hs=0.65, base=b * 1.1)


LUIGI14 = dict(
    T=210, S=330, M=480, margin=0.03, depth=0.45, gold="rosato", wear=0.5, patina=0.8, seed=31,
    profile=[P(.05, "qr", 0, .5), P(.10, "lin", .5, .5, b=.10, name="ovoli"), P(.06, "cove", .5, .38),
             P(.36, "lin", .38, .42, b=.26, name="acanto"), P(.05, "lin", .36, .36), P(.14, "ogee", .36, .16),
             P(.10, "lin", .20, .20, b=.05, name="punte"), P(.06, "lin", .18, .0)],
    motif=l14_motif, corner=l14_corner, center=l14_center, Lc=400,
)


# ------------------------------------------------------------ 4. "Salvator Rosa" a gola, intaglio leggero, brunito
def sal_motif(U, vp, r, T, D):
    add = mo.gadroons(U, vp, r["bacc"], 40, 0.8, tilt=0.8) + mo.leaf_tips(U, vp, r["punte"], 48, 0.55) \
        + mo.flutes(U, vp, r["gola"], 60, 0.25)
    return add, None


def sal_corner(c, T, S, D):
    b = 0.45 * D
    acanthus(c, (0.30 * T, 0.30 * T), 0.25 * PI, 0.62 * T, 0.30 * T, curl=0, tipcurl=0, lobes=4, hs=0.4, base=b)


SALVATOR = dict(
    T=170, S=260, M=480, margin=0.0, depth=0.55, gold="brunito", wear=0.45, patina=0.9, seed=41,
    profile=[P(.08, "qr", 0, .85), P(.08, "lin", .85, .85, b=.08, name="bacc"), P(.50, "cove", .85, .18, name="gola"),
             P(.10, "lin", .18, .24, b=.06, name="punte"), P(.08, "lin", .2, .0)],
    motif=sal_motif, corner=sal_corner,
)


# ------------------------------------------------------------ 5. Fiorentina a fogliame continuo, oro antico
def fio_motif(U, vp, r, T, D):
    add = mo.pearls_and_reels(U, vp, r["perle"], 48, 0.9) + mo.foliage(U, vp, r["fogliame"], 160, 0.6) \
        + mo.beads(U, vp, r["perline"], 24, 0.9)
    return add, None


def fio_corner(c, T, S, D):
    b = 0.5 * D
    rosette(c, (0.46 * T, 0.46 * T), 0.26 * T, 8, hs=0.55, base=b)
    rosette(c, (0.46 * T, 0.46 * T), 0.13 * T, 6, hs=0.8, base=b * 1.15, rot=0.4)


FIORENTINA = dict(
    T=200, S=300, M=480, margin=0.0, depth=0.45, gold="antico", wear=0.95, patina=0.85, seed=51,
    profile=[P(.05, "qr", 0, .5), P(.07, "lin", .5, .5, b=.1, name="perle"), P(.05, "cove", .5, .4),
             P(.55, "lin", .4, .4, b=.25, name="fogliame"), P(.06, "cove", .4, .2),
             P(.07, "lin", .22, .22, b=.06, name="perline"), P(.05, "lin", .2, 0)],
    motif=fio_motif, corner=fio_corner,
)


# ------------------------------------------------------------ 6. Olandese ebanizzata con battuta e fregi dorati
def ola_motif(U, vp, r, T, D):
    add = mo.ripple(U, vp, r["onde1"], 20, 1.0) + mo.ripple(U, vp, r["onde2"], 24, 1.0) \
        + mo.ripple(U, vp, r["onde3"], 16, 1.0) + mo.beads(U, vp, r["battuta"], 20, 0.9)
    return add, None


def ola_corner(c, T, S, D):
    b = 0.42 * D
    acanthus(c, (0.40 * T, 0.40 * T), -0.75 * PI + PI, 0.70 * T, 0.40 * T, curl=0, tipcurl=0, lobes=4, hs=0.35, base=b)
    acanthus(c, (0.45 * T, 0.30 * T), 0.0, 0.60 * T, 0.26 * T, curl=0.2, tipcurl=1.0, lobes=3, hs=0.35, base=b)
    scroll(c, (0.30 * T, 0.30 * T), 0.10 * T, 0.03 * T, PI * 1.25, PI * 3.0, 0.05 * T, 0.025 * T, hs=0.8, base=b)


def ola_center(c, T, S, D):
    b = 0.42 * D
    acanthus(c, (0.10 * T, 0.48 * T), 0.0, 0.62 * T, 0.30 * T, curl=-0.2, tipcurl=-1.0, lobes=3, hs=0.35, base=b)
    cartouche(c, (0, 0.48 * T), 0.20 * T, 0.16 * T, 0.05 * T, hs=0.8, base=b)
    rosette(c, (0, 0.48 * T), 0.08 * T, 6, hs=0.7, base=b * 1.05)


OLANDESE = dict(
    T=200, S=330, M=480, margin=0.0, depth=0.4, gold="chiaro", wear=0.3, patina=0.7, seed=61, orn_mat=0,
    profile=[P(.06, "qr", 0, .45, mat=1), P(.10, "lin", .45, .45, b=.03, mat=1, name="onde1"),
             P(.06, "lin", .45, .36, mat=1), P(.14, "lin", .36, .36, b=.04, mat=1, name="onde2"),
             P(.30, "lin", .36, .18, mat=1), P(.07, "lin", .18, .2, b=.02, mat=1, name="onde3"),
             P(.07, "lin", .20, .20, b=.08, mat=0, name="battuta"), P(.06, "lin", .16, .0, mat=0)],
    motif=ola_motif, corner=ola_corner, center=ola_center, Lc=320,
)


# ------------------------------------------------------------ 7. Cassetta dorata con canaletta e perline
def cas_motif(U, vp, r, T, D):
    add = mo.beads(U, vp, r["perle"], 24, 0.9) + mo.pearls_and_reels(U, vp, r["perline"], 40, 0.9) \
        + mo.punch(U, vp, r["fregio"], 8, 0.5)
    return add, None


def cas_corner(c, T, S, D):
    b = 0.32 * D
    rosette(c, (0.49 * T, 0.49 * T), 0.17 * T, 6, hs=0.55, base=b)
    for k in range(4):
        a = PI / 4 + k * PI / 2
        acanthus(c, (0.49 * T + math.cos(a) * 0.12 * T, 0.49 * T + math.sin(a) * 0.12 * T), a, 0.16 * T, 0.10 * T,
                 lobes=2, hs=0.4, base=b * 0.9)


def cas_center(c, T, S, D):
    b = 0.32 * D
    rosette(c, (0, 0.49 * T), 0.15 * T, 6, hs=0.55, base=b)
    acanthus(c, (0.12 * T, 0.49 * T), 0.0, 0.40 * T, 0.16 * T, curl=0.15, tipcurl=0.8, lobes=3, hs=0.4, base=b * 0.9)


CASSETTA = dict(
    T=180, S=270, M=480, margin=0.0, depth=0.42, gold="antico", wear=0.6, patina=0.75, seed=71, exposure=1.05,
    profile=[P(.05, "qr", 0, .6), P(.07, "lin", .6, .6, b=.1, name="perle"), P(.05, "lin", .6, .5),
             P(.06, "cove", .5, .25), P(.40, "lin", .25, .25, b=-.06, name="fregio"), P(.06, "cove", .25, .5),
             P(.08, "lin", .5, .5, b=.1, name="perline"), P(.07, "lin", .5, .4), P(.06, "lin", .4, 0)],
    motif=cas_motif, corner=cas_corner, center=cas_center, Lc=280,
)


# ------------------------------------------------------------ 8. Ottocentesca in pastiglia con fascia in lino
def ott_motif(U, vp, r, T, D):
    add = mo.palmettes(U, vp, r["palm"], 80, 0.7) + mo.beads(U, vp, r["perle"], 16, 0.9) \
        + mo.beads(U, vp, r["slip"], 20, 0.9)
    return add, None


def ott_corner(c, T, S, D):
    b = 0.5 * D
    scroll(c, (0.36 * T, 0.18 * T), 0.10 * T, 0.03 * T, PI * 0.8, PI * 2.6, 0.045 * T, 0.022 * T, hs=0.8, base=b)
    acanthus(c, (0.20 * T, 0.20 * T), 0.25 * PI, 0.40 * T, 0.24 * T, lobes=3, hs=0.45, base=b)
    acanthus(c, (0.42 * T, 0.12 * T), 0.05, 0.42 * T, 0.16 * T, curl=0.2, tipcurl=0.9, lobes=3, hs=0.45, base=b)
    shell(c, (0.34 * T, 0.34 * T), -0.75 * PI, 0.26 * T, 0.9, 9, hs=0.45, base=b)


def ott_center(c, T, S, D):
    b = 0.5 * D
    shell(c, (0, 0.40 * T), -PI / 2, 0.34 * T, 1.0, 11, hs=0.45, base=b)
    scroll(c, (0.22 * T, 0.16 * T), 0.09 * T, 0.025 * T, PI * 1.0, PI * 2.8, 0.04 * T, 0.02 * T, hs=0.8, base=b)
    acanthus(c, (0.26 * T, 0.24 * T), 0.0, 0.36 * T, 0.14 * T, curl=0.2, tipcurl=0.8, lobes=3, hs=0.45, base=b)


OTTOCENTESCA = dict(
    T=230, S=340, M=480, margin=0.0, depth=0.35, gold="chiaro", wear=0.25, patina=0.7, seed=81,
    profile=[P(.05, "qr", 0, .55), P(.16, "lin", .55, .55, b=.16, name="palm"), P(.05, "cove", .55, .40),
             P(.05, "lin", .42, .42, b=.05, name="perle"), P(.06, "cove", .40, .22), P(.38, "lin", .22, .14, mat=2),
             P(.06, "lin", .16, .16, b=.07, name="slip"), P(.05, "lin", .14, 0)],
    motif=ott_motif, corner=ott_corner, center=ott_center, Lc=300,
)

import math  # noqa

STILI = {"luigi15": LUIGI15, "barocca": BAROCCA, "luigi14": LUIGI14, "salvator": SALVATOR, "fiorentina": FIORENTINA,
         "olandese": OLANDESE, "cassetta": CASSETTA, "ottocentesca": OTTOCENTESCA}
