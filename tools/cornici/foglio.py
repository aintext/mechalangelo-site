# Foglio di confronto: angolo + meta' lato di ogni cornice, su parete, con dipinto.
import sys, os, json
from PIL import Image
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "img")
out = sys.argv[1]; scale = float(sys.argv[2]); names = sys.argv[3:]
meta = json.load(open(os.path.join(d, "cornici", "cornici.json")))
tiles = []
for n in names:
    m = meta[n]
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anteprime", f"_{n}-full.webp")
    if not os.path.exists(p): p = os.path.join(d, "cornici", f"{n}.webp")
    fr = Image.open(p).convert("RGBA"); N, T = m["N"], m["T"]
    bg = Image.new("RGBA", (N, N), (240, 236, 229, 255))
    art = Image.open(os.path.join(d, "bacio.jpg")).convert("RGBA").resize((N - 2 * T + 8, N - 2 * T + 8))
    bg.paste(art, (T - 4, T - 4)); bg.alpha_composite(fr)
    q = bg.crop((0, 0, N // 2 + 40, N // 2 + 40))
    q = q.resize((int(q.width * scale), int(q.height * scale)), Image.LANCZOS)
    tiles.append(q)
cols = 4; w = max(t.width for t in tiles); h = max(t.height for t in tiles)
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (w + 10), rows * (h + 10)), (240, 236, 229))
for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * (w + 10), (i // cols) * (h + 10)))
sheet.save(out)
