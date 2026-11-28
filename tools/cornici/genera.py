# Genera le cornici in img/cornici/ e stampa i parametri per FR (index.html).
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from rendi import write
from stili import STILI
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "cornici")
sel = sys.argv[1:] or list(STILI)
meta = {}
mp = os.path.join(out, "cornici.json")
if os.path.exists(mp):
    meta = json.load(open(mp))
for k in sel:
    t = time.time()
    meta[k] = write(STILI[k], k, out)
    print(k, meta[k], f"{time.time()-t:.1f}s", flush=True)
json.dump(meta, open(mp, "w"), indent=1)
