# Anteprima: cornice completa su parete con un dipinto dentro.
import sys, os, json
from PIL import Image
d = os.path.join(os.path.dirname(__file__), "..", "..", "img")
names = sys.argv[1:]
tiles = []
meta = json.load(open(os.path.join(d, "cornici", "cornici.json")))
for n in names:
    m = meta[n]
    p = os.path.join(d, "cornici", f"_{n}-full.webp")
    if not os.path.exists(p): p = os.path.join(d, "cornici", f"{n}.webp")
    fr = Image.open(p).convert("RGBA")
    N, T = m["N"], m["T"]
    bg = Image.new("RGBA", (N + 80, N + 80), (240, 236, 229, 255))
    art = Image.open(os.path.join(d, "creazione.jpg")).convert("RGBA").resize((N - 2 * T + 8, N - 2 * T + 8))
    bg.paste(art, (40 + T - 4, 40 + T - 4))
    bg.alpha_composite(fr, (40, 40))
    tiles.append(bg)
W = sum(t.width for t in tiles); H = max(t.height for t in tiles)
out = Image.new("RGB", (W, H), (240, 236, 229))
x = 0
for t in tiles: out.paste(t, (x, 0)); x += t.width
out.save(sys.argv[0].replace("anteprima.py", "") + "../../../prev.png") if False else out.save("/tmp/claude-0/-home-claude-mechalangelo-site/3599fe68-3578-5199-803e-9d29035ebb19/scratchpad/prev.png")
