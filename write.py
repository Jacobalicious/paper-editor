"""Step 3: write the correct answer where the wrong one was, in the writer's hand.

Usage:  python write.py samples/example-2plus2.png
Needs out/<name>-graded.json (grade.py + erase.py) and out/<name>-erased.png.

How: Claude looks at the writer's line of handwriting and sketches the new answer
as pen strokes in that style. We then draw those strokes with the writer's
measured stroke width, pencil darkness, grain and blur, and blend them onto the
erased paper so the paper texture shows through the ink.

Saves out/<name>-corrected.png and out/<name>-compare.png (zoomed before/after).
"""

import base64
import json
import sys
from pathlib import Path

import anthropic
import cv2
import numpy as np
from dotenv import load_dotenv

from erase import ink_mask, paper_background
from grade import MODEL

SS = 4  # supersampling factor for drawing; anti-aliases like a camera would
PEN_SCALE = 0.85  # measured width includes blurred edges; the pen itself is a bit thinner

PROMPT = """The image shows one line of someone's handwriting, enlarged. The part
marked by the red box is the wrong answer "{old}". It will be erased and replaced with
"{new}", written by the same person.

Study how this person writes: stroke shapes, how they start and end strokes, slant,
proportions, how wide characters are relative to their height, spacing, whether
strokes are lifted or joined, small wobbles. Then write "{new}" the way THEY would,
as pen strokes.

Coordinates: y=0 is the top of the replacement and y=100 is its bottom (the same height
as "{old}"; the box shows that height). x starts at 0 on the left, in the same units
(so a character as wide as it is tall spans x 0-100).
Each stroke is one continuous pen movement, in drawing order, as points along the pen's
path. Use enough points to capture curves smoothly (roughly every 3-5 units). Keep the
natural irregularity of real handwriting; do not make it geometrically perfect."""

SCHEMA = {
    "type": "object",
    "properties": {
        "style_notes": {"type": "string"},
        "strokes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    }
                },
                "required": ["points"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["style_notes", "strokes"],
    "additionalProperties": False,
}


# ---------- measuring the writer's pen ----------

def measure_pen(img, strong, box):
    """Stroke width (px) and core ink colour of the old answer."""
    x0, y0, x1, y1 = box
    m = strong[y0:y1, x0:x1]
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)))
    width = float(2 * np.median(dt[ridge])) if ridge.any() else 2.0

    crop = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    g = crop.mean(axis=1)
    inked = m.reshape(-1) > 0
    darkest = g[inked] <= np.percentile(g[inked], 10)
    core = crop[inked][darkest].mean(axis=0)
    return width, core


def edge_softness(gray, bg, strong):
    """Rough blur of the photo: how wide the ink edges are. Returns a Gaussian sigma."""
    diff = cv2.subtract(bg, gray).astype(np.float32)
    edges = cv2.dilate(strong, np.ones((3, 3), np.uint8)) & ~cv2.erode(strong, np.ones((3, 3), np.uint8))
    if not edges.any():
        return 0.7
    gx = cv2.Sobel(diff, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(diff, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)[edges > 0]
    peak = np.percentile(diff[strong > 0], 75)
    # A sharper edge has a higher gradient for the same contrast.
    ratio = peak / max(np.percentile(grad, 75) / 4, 1e-3)
    return float(np.clip(0.4 * ratio, 0.3, 1.5))


# ---------- asking Claude for strokes ----------

def line_image(img, box):
    x0, y0, x1, y1 = box
    h = y1 - y0
    W = img.shape[1]
    ly0, ly1 = max(0, y0 - h // 2), min(img.shape[0], y1 + h // 2)
    crop = img[ly0:ly1, 0:W].copy()
    cv2.rectangle(crop, (x0 - 2, y0 - ly0 - 2), (x1 + 2, y1 - ly0 + 2), (0, 0, 220), 1)
    scale = max(1, 900 // crop.shape[1] + 1)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", crop)
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def ask_strokes(img, box, old, new):
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": line_image(img, box)}},
            {"type": "text", "text": PROMPT.format(old=old, new=new)},
        ]}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to draw the strokes.")
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    strokes = [np.array([p[:2] for p in s["points"]], np.float32)
               for s in data["strokes"] if len(s["points"]) >= 2]
    return strokes, data["style_notes"]


# ---------- drawing ----------

def catmull_rom(pts, step):
    """Smooth a polyline through its points, sampled every `step` units."""
    if len(pts) < 3:
        n = max(2, int(np.linalg.norm(pts[-1] - pts[0]) / step))
        return np.linspace(pts[0], pts[-1], n)
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(p) - 2):
        a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
        n = max(2, int(np.linalg.norm(c - b) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2
                          + (-a + 3 * b - 3 * c + d) * t ** 3))
    out.append(p[-2][None])
    return np.vstack(out)


def render(strokes, unit_px, pen_px, rng):
    """Draw strokes into a coverage map (0..1) at final resolution."""
    allp = np.vstack(strokes)
    margin = pen_px * 3
    minx = min(0, allp[:, 0].min())
    w = int((allp[:, 0].max() - minx) * unit_px + 2 * margin) + 1
    h = int(100 * unit_px + 2 * margin) + 1
    canvas = np.zeros((h * SS, w * SS), np.float32)
    ox = margin - minx * unit_px
    for s in strokes:
        px = catmull_rom(s * unit_px, 0.25)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(px, axis=0), axis=1))]
        t = seg / max(seg[-1], 1e-3)
        pressure = np.minimum(1, 0.7 + t / 0.12 * 0.3) * np.minimum(1, 0.8 + (1 - t) / 0.15 * 0.2)
        pressure = pressure * np.interp(t, np.linspace(0, 1, 6), rng.normal(1, 0.08, 6))
        for (x, y), pr in zip(px, pressure):
            r = max(1, int(round(pen_px / 2 * pr * SS)))
            c = (int((x + ox) * SS), int((y + margin) * SS))
            layer = canvas[max(0, c[1] - r - 1):c[1] + r + 2, max(0, c[0] - r - 1):c[0] + r + 2]
            if layer.size == 0:
                continue
            stamp = np.zeros_like(layer)
            cv2.circle(stamp, (c[0] - max(0, c[0] - r - 1), c[1] - max(0, c[1] - r - 1)),
                       r, float(pr), -1, cv2.LINE_AA)
            np.maximum(layer, stamp, out=layer)
    cov = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    grain = cv2.GaussianBlur(rng.normal(0, 1, cov.shape).astype(np.float32), (0, 0), 0.8)
    cov = cov * np.clip(1 + 0.25 * grain, 0.5, 1.3)
    return np.clip(cov, 0, 1), int(round(ox)), int(round(margin))


def blend(page, cov, top_left, core, paper, blur_sigma):
    """Multiply-blend ink onto the page: paper texture keeps showing through."""
    peak = np.percentile(cov[cov > 0.05], 90) if (cov > 0.05).any() else 1
    cov = cv2.GaussianBlur(cov, (0, 0), blur_sigma)
    # Blur spreads thin strokes out and lightens them; restore the centre darkness.
    cov = np.clip(cov * peak / max(np.percentile(cov[cov > 0.05], 90), 1e-3), 0, 1)
    x, y = top_left
    h, w = cov.shape
    region = page[y:y + h, x:x + w].astype(np.float32)
    cov = cov[:region.shape[0], :region.shape[1], None]
    ratio = np.clip(core / np.maximum(paper, 1), 0, 1)
    region = region * (1 - cov + cov * ratio)
    page[y:y + h, x:x + w] = np.clip(region, 0, 255).astype(np.uint8)


def compare_strip(before, after, box, scale=4):
    x0, y0, x1, y1 = box
    pad = (y1 - y0) // 2
    sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - 3 * pad), x1 + pad))
    a = cv2.resize(before[sl], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(after[sl], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return np.hstack([a, np.full((a.shape[0], 12, 3), 255, np.uint8), b])


def main():
    args = [a for a in sys.argv[1:] if a != "--redraw"]
    redraw = "--redraw" in sys.argv
    if len(args) != 1:
        sys.exit("Usage: python write.py <photo> [--redraw]")
    load_dotenv()
    src = Path(args[0])
    out_dir = Path("out")
    graded = json.loads((out_dir / f"{src.stem}-graded.json").read_text())

    original = cv2.imread(str(src), cv2.IMREAD_COLOR)
    page = cv2.imread(str(out_dir / f"{src.stem}-erased.png"), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    bg = paper_background(gray)
    strong, _ = ink_mask(gray, bg)
    blur_sigma = edge_softness(gray, bg, strong)
    rng = np.random.default_rng(1)

    strips = []
    for p in graded["to_fix"]:
        if "ink_box" not in p:
            continue
        box = p["ink_box"]
        pen_px, core = measure_pen(original, strong, box)
        x0, y0, x1, y1 = box
        orig_paper = cv2.merge([paper_background(original[..., c]) for c in range(3)])
        page_paper = cv2.merge([paper_background(page[..., c]) for c in range(3)])
        op = orig_paper[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32).mean(axis=0)
        paper = page_paper[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32).mean(axis=0)
        ink = core / np.maximum(op, 1) * paper  # same darkening as the real pen

        if "strokes" in p and not redraw:
            strokes = [np.array(s, np.float32) for s in p["strokes"]]
        else:
            strokes, notes = ask_strokes(original, box, p["written_answer"], p["correct_answer"])
            p["strokes"] = [s.tolist() for s in strokes]
            print(f"Writing '{p['correct_answer']}' - style: {notes}")
        # Start the new answer where the old ink started, so spacing after "=" is kept.
        left = min(st[:, 0].min() for st in strokes)
        strokes = [st - [left, 0] for st in strokes]
        cov, ox, oy = render(strokes, (y1 - y0) / 100, pen_px * PEN_SCALE, rng)
        blend(page, cov, (x0 - ox, y0 - oy), ink, paper, blur_sigma)
        strips.append(compare_strip(original, page, box))
        print(f"  pen width {pen_px:.1f}px, blur {blur_sigma:.2f}")
    (out_dir / f"{src.stem}-graded.json").write_text(json.dumps(graded, indent=2))

    cv2.imwrite(str(out_dir / f"{src.stem}-corrected.png"), page)
    if strips:
        w = max(s.shape[1] for s in strips)
        strips = [np.pad(s, ((0, 8), (0, w - s.shape[1]), (0, 0)), constant_values=255) for s in strips]
        cv2.imwrite(str(out_dir / f"{src.stem}-compare.png"), np.vstack(strips))
    print(f"Saved out/{src.stem}-corrected.png and out/{src.stem}-compare.png")


if __name__ == "__main__":
    main()
