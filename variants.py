"""Make a batch of different versions of the corrected answer and a web page to compare them.

Usage:  python variants.py samples/example-2plus2.png [sketches]
Needs grade.py + erase.py to have run on the photo first (run.ps1 does that).

Each round asks Claude for `sketches` different drawings of the answer (default 4)
and renders each with a few pen settings, so one round = sketches x len(STYLES)
variants. Rounds are kept in docs/rounds/<time>/ and docs/index.html shows them all,
newest first, so earlier favourites stay visible.
"""

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from write import DEFAULT_STYLE, get_strokes, load_page, write_answer

STYLES = {
    "as measured": {},
    "thinner + darker": {"width": 0.85, "dark": 1.12},
    "rougher": {"wobble": 1.7, "grain": 1.5},
}
DOCS = Path("docs")
ZOOM = 3


def zoom_crop(img, boxes):
    """Close-up around the answers, with some of the line to the left for context."""
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    h = y1 - y0
    crop = img[max(0, y0 - h // 2):y1 + h // 2, max(0, x0 - 2 * h):x1 + h // 2]
    return cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)


def save_jpg(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def make_round(src, n_sketches):
    ctx = load_page(src)
    fixes = [p for p in ctx["graded"]["to_fix"] if "ink_box" in p]
    if not fixes:
        sys.exit("Nothing to fix in this photo (run run.ps1 on it first).")
    boxes = [p["ink_box"] for p in fixes]

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rdir = DOCS / "rounds" / stamp
    save_jpg(rdir / "original-zoom.jpg", zoom_crop(ctx["original"], boxes))
    save_jpg(rdir / "original.jpg", ctx["original"])

    variants = []
    for k in range(n_sketches):
        print(f"Sketch {k + 1} of {n_sketches}...")
        sketches = [get_strokes(ctx, p, redraw=True) for p in fixes]
        for name, tweaks in STYLES.items():
            style = {**DEFAULT_STYLE, **tweaks}
            page = ctx["erased"].copy()
            rng = np.random.default_rng(len(variants))
            for p, strokes in zip(fixes, sketches):
                write_answer(ctx, page, p, strokes, rng, style)
            num = len(variants) + 1
            save_jpg(rdir / f"v{num:02d}-zoom.jpg", zoom_crop(page, boxes))
            save_jpg(rdir / f"v{num:02d}.jpg", page)
            variants.append({"num": num, "sketch": k + 1, "style": name,
                             "strokes": [[s.tolist() for s in sk] for sk in sketches]})

    meta = {"photo": src.name, "time": stamp,
            "answers": [f"{p['written_answer']} -> {p['correct_answer']}" for p in fixes],
            "variants": variants}
    (rdir / "round.json").write_text(json.dumps(meta, indent=1))
    return rdir


def build_index():
    rounds = sorted((DOCS / "rounds").glob("*/round.json"), reverse=True)
    sections = []
    for rj in rounds:
        meta = json.loads(rj.read_text())
        base = f"rounds/{meta['time']}"
        cards = [f'''<figure class="card ref">
  <a href="{base}/original.jpg"><img src="{base}/original-zoom.jpg" alt="Original"></a>
  <figcaption><b>Real</b> original photo</figcaption></figure>''']
        for v in meta["variants"]:
            cards.append(f'''<figure class="card" id="{meta['time']}-{v['num']}">
  <a href="{base}/v{v['num']:02d}.jpg"><img loading="lazy" src="{base}/v{v['num']:02d}-zoom.jpg" alt="Variant {v['num']}"></a>
  <figcaption><b>#{v['num']}</b> sketch {v['sketch']} &middot; {html.escape(v['style'])}</figcaption></figure>''')
        when = datetime.strptime(meta["time"], "%Y-%m-%d_%H%M%S").strftime("%b %d, %H:%M")
        sections.append(f'''<section>
<h2>Round {when} <span>{html.escape(meta['photo'])} &middot; {html.escape(", ".join(meta['answers']))}</span></h2>
<div class="grid">{"".join(cards)}</div></section>''')

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Editor Variants</title>
<style>
:root {{ --bg:#f6f4ef; --fg:#1d1c1a; --muted:#6b675f; --card:#fff; --line:#e3dfd6; --accent:#b4471f; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161514; --fg:#ecebe7; --muted:#9b978e; --card:#22201e; --line:#34312d; --accent:#e0794f; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
header, section {{ max-width:1200px; margin:0 auto; padding:16px; }}
h1 {{ margin:16px 0 4px; font-size:22px; }}
header p {{ margin:0; color:var(--muted); }}
h2 {{ font-size:16px; margin:24px 0 12px; }}
h2 span {{ font-weight:400; color:var(--muted); margin-left:8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:12px; }}
.card {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.card.ref {{ border-color:var(--accent); }}
.card img {{ display:block; width:100%; height:auto; }}
figcaption {{ padding:8px 10px; font-size:14px; color:var(--muted); }}
figcaption b {{ color:var(--fg); margin-right:4px; }}
.ref figcaption b {{ color:var(--accent); }}
</style></head><body>
<header><h1>Paper editor: variants</h1>
<p>Each card is a close-up of one version of the corrected answer. The orange card is the real, untouched photo for comparison. Click any card for the full page. Tell Claude the number of your favourite.</p></header>
{"".join(sections) or "<section><p>No rounds yet.</p></section>"}
</body></html>'''
    (DOCS / "index.html").write_text(page, encoding="utf-8")


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python variants.py <photo> [sketches]")
    load_dotenv()
    n = int(sys.argv[2]) if len(sys.argv) == 3 else 4
    rdir = make_round(Path(sys.argv[1]), n)
    build_index()
    print(f"Saved {rdir} and docs/index.html")


if __name__ == "__main__":
    main()
