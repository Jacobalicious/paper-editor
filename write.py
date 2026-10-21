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
import math
import sys
from pathlib import Path

import anthropic
import cv2
import numpy as np
from dotenv import load_dotenv

from erase import ink_mask, paper_background
from grade import MODEL

SS = 4  # supersampling factor for drawing; anti-aliases like a camera would
MIN_SHRINK = 0.55  # a cramped answer gets written smaller, but only down to this

# Rendering knobs. 1.0 = what was measured from the writer's own strokes.
DEFAULT_STYLE = {"width": 1.0, "dark": 1.0, "wobble": 1.0, "grain": 1.0, "blur": 1.0}

PROMPT = """The image shows one line of someone's handwriting, enlarged. The part
marked by the red box is the wrong answer "{old}". It will be erased and replaced with
"{new}", written by the same person.

Study how this person writes: stroke shapes, how they start and end strokes, slant,
proportions, how wide characters are relative to their height, spacing, whether
strokes are lifted or joined, small wobbles. Then write "{new}" the way THEY would,
as pen strokes.

Coordinates: y=0 is the top of the replacement and y=100 is its bottom (the same height
as "{old}"; the box shows that height). Keep the writing inside that height - it must
sit on the same line as everything around it, not tower over it. x starts at 0 on the
left, in the same units (so a character as wide as it is tall spans x 0-100).
There are only {room} units of blank paper before the next thing on this line, so the
whole of "{new}" has to fit in that width. If it is longer than "{old}", pack the
characters closer together, the way this person would when running out of room.
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

def measure_pen(img, gray, bg, strong, box, sigma):
    """Work out the real pen: its width (px) and true darkness.

    In the photo a thin stroke is smeared by blur, so it looks wider and lighter
    than the pencil line really is. We measure two things the blur can't change
    much - total ink per pixel of stroke length, and the darkest point across a
    stroke - and solve for the width and darkness that, once blurred, give both.
    Returns (width_px, darkness_gain) where gain scales the observed peak darkness
    up to the true pencil darkness.
    """
    x0, y0, x1, y1 = box
    m = strong[y0:y1, x0:x1]
    # A thin stroke's outline goes up one side and back down the other, so the
    # stroke length is about half the total outline length.
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    length = max(sum(cv2.arcLength(c, True) for c in contours) / 2, 1)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    centre = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)))
    diff = cv2.subtract(bg, gray)[y0:y1, x0:x1].astype(np.float32)
    near = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    mass = diff[near].sum() / length          # ink per pixel of length
    peak = np.percentile(diff[centre], 75)    # observed darkness at stroke centres

    k = 2 * math.sqrt(2) * max(sigma, 1e-3)
    def seen_peak(w):
        return mass / w * math.erf(w / k)
    lo, hi = 0.5, 12.0
    if seen_peak(lo) < peak:                  # sharper than our blur estimate
        width = mass / peak
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if seen_peak(mid) > peak else (lo, mid)
        width = (lo + hi) / 2
    gain = (mass / width) / peak
    return float(width), float(gain)


def ink_colour(img, strong, box):
    """Colour of the darkest part of the old strokes."""
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    inked = strong[y0:y1, x0:x1].reshape(-1) > 0
    g = crop[inked].mean(axis=1)
    return crop[inked][g <= np.percentile(g, 15)].mean(axis=0)


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


def ask_strokes(img, box, old, new, room):
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
            {"type": "text", "text": PROMPT.format(old=old, new=new, room=room) + favorites_hint()},
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


def render(strokes, unit_px, pen_px, rng, style=DEFAULT_STYLE):
    """Draw strokes into a coverage map (0..1) at final resolution."""
    allp = np.vstack(strokes)
    margin = pen_px * 3
    minx = min(0, allp[:, 0].min())
    miny = min(0, allp[:, 1].min())            # room for anything above the line
    maxy = max(100, allp[:, 1].max())          # ...and for descenders below it
    w = int((allp[:, 0].max() - minx) * unit_px + 2 * margin) + 1
    h = int((maxy - miny) * unit_px + 2 * margin) + 1
    canvas = np.zeros((h * SS, w * SS), np.float32)
    ox = margin - minx * unit_px
    margin = margin - miny * unit_px           # where y=0 ends up on the canvas
    for s in strokes:
        px = catmull_rom(s * unit_px, 0.25)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(px, axis=0), axis=1))]
        t = seg / max(seg[-1], 1e-3)
        px = px + hand_tremor(px, seg, pen_px, rng) * style["wobble"]
        pressure = np.minimum(1, 0.75 + t / 0.12 * 0.25) * np.minimum(1, 0.85 + (1 - t) / 0.15 * 0.15)
        pressure = pressure * np.interp(t, np.linspace(0, 1, 6), rng.normal(1, 0.07, 6))
        pressure = pressure / pressure.mean()  # average pressure = the measured pen
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
    canvas = np.clip(canvas, 0, 1)
    # Graphite grain: paper fibres catch the lead unevenly. Raising coverage to a
    # random power leaves solid centres alone but makes the soft edges ragged.
    fine = cv2.GaussianBlur(rng.normal(0, 1, canvas.shape).astype(np.float32), (0, 0), SS * 0.4)
    fine /= fine.std() + 1e-6
    g = style["grain"]
    canvas = canvas ** np.clip(1 + 0.6 * g * fine, 0.4, 2.5) * np.clip(1 + 0.08 * g * fine, 0.8, 1.2)
    cov = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(cov, 0, 1), int(round(ox)), int(round(margin))


def hand_tremor(px, seg, pen_px, rng):
    """Small sideways wobble along a stroke - hands never draw perfectly straight."""
    if len(px) < 3:
        return np.zeros_like(px)
    d = np.gradient(px, axis=0)
    normal = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    off = np.zeros(len(px))
    for period, amp in ((40, 0.45), (14, 0.25), (6, 0.12)):
        off += amp * np.sin(2 * np.pi * seg / period + rng.uniform(0, 2 * np.pi))
    return normal * (off * max(pen_px / 2.8, 0.6))[:, None]


def blend(page, cov, top_left, core, paper, blur_sigma):
    """Multiply-blend ink onto the page: paper texture keeps showing through."""
    cov = cv2.GaussianBlur(cov, (0, 0), blur_sigma)
    x, y = top_left
    # Keep it on the page: trim anything that would fall off an edge.
    cov = cov[max(0, -y):, max(0, -x):]
    x, y = max(0, x), max(0, y)
    h, w = cov.shape
    region = page[y:y + h, x:x + w].astype(np.float32)
    if region.size == 0:
        return
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


def load_page(src):
    """Everything about the photo that every variant shares."""
    out_dir = Path("out")
    ctx = {"src": src, "out_dir": out_dir,
           "graded": json.loads((out_dir / f"{src.stem}-graded.json").read_text()),
           "original": cv2.imread(str(src), cv2.IMREAD_COLOR),
           "erased": cv2.imread(str(out_dir / f"{src.stem}-erased.png"), cv2.IMREAD_COLOR)}
    gray = cv2.cvtColor(ctx["original"], cv2.COLOR_BGR2GRAY)
    bg = paper_background(gray)
    strong, _ = ink_mask(gray, bg)
    ctx.update(gray=gray, bg=bg, strong=strong, blur_sigma=edge_softness(gray, bg, strong),
               orig_paper=cv2.merge([paper_background(ctx["original"][..., c]) for c in range(3)]),
               page_paper=cv2.merge([paper_background(ctx["erased"][..., c]) for c in range(3)]))
    return ctx


def load_favorites():
    """Hand-picked drawings (favorites.json): answer text -> strokes."""
    f = Path(__file__).parent / "favorites.json"
    return json.loads(f.read_text()) if f.exists() else {}


def favorites_hint():
    favs = load_favorites()
    if not favs:
        return ""
    examples = "\n".join(f'"{k}": {json.dumps([[[round(x), round(y)] for x, y in s] for s in v])}'
                         for k, v in favs.items())
    return ("\n\nThese drawings, in the same coordinates, were judged to look most like this "
            "writer. Draw any of these characters the same way:\n" + examples)


def vary(strokes, rng):
    """A favorite, redrawn slightly differently: small shifts, tilt and size change."""
    angle = rng.normal(0, 0.03)
    c, s = np.cos(angle), np.sin(angle)
    scale = rng.normal(1, 0.03, 2)
    out = []
    for st in strokes:
        st = np.array(st, np.float32)
        st = (st - 50) @ np.array([[c, -s], [s, c]], np.float32).T * scale + 50
        # Gentle bend along the stroke: random offsets at a few points, smoothly joined.
        t = np.linspace(0, 1, len(st))
        bend = np.stack([np.interp(t, np.linspace(0, 1, 4), rng.normal(0, 0.8, 4)) for _ in range(2)], 1)
        out.append(st + rng.normal(0, 1.0, 2) + bend)
    return out


def get_strokes(ctx, p, redraw):
    if "strokes" in p and not redraw:
        return [np.array(s, np.float32) for s in p["strokes"]]
    fav = load_favorites().get(p["correct_answer"].strip())
    if fav:
        strokes = vary(fav, np.random.default_rng())
        p["strokes"] = [s.tolist() for s in strokes]
        print(f"Writing '{p['correct_answer']}' from your favorite drawing")
        return strokes
    box = p["ink_box"]
    room = int(room_to_write(ctx, box) / max(box[3] - box[1], 1) * 100)
    strokes, notes = ask_strokes(ctx["original"], box, p["written_answer"],
                                 p["correct_answer"], room)
    p["strokes"] = [s.tolist() for s in strokes]
    print(f"Writing '{p['correct_answer']}' - style: {notes}")
    return strokes


def room_to_write(ctx, box):
    """Blank paper to the right of the old answer, in pixels.

    Scans along the old answer's own line for the next ink; if there is none,
    the writing can run to the edge of the photo (less a small margin).
    """
    x0, y0, x1, y1 = box
    h = max(y1 - y0, 1)
    W = ctx["strong"].shape[1]
    band = ctx["strong"][max(0, y0 - h // 3):y1 + h // 3, :] > 0
    cols = band.sum(axis=0)
    gap = max(2, h // 8)                       # ignore the old answer's own smear
    ahead = np.nonzero(cols[x1 + gap:] >= 2)[0]
    right = (x1 + gap + int(ahead[0])) if len(ahead) else W - h // 4
    return max(right - max(2, h // 6) - x0, h)  # never squeeze below one character


def write_answer(ctx, page, p, strokes, rng, style=DEFAULT_STYLE):
    """Draw one answer onto `page` (in place). Returns the measured pen settings."""
    box = p["ink_box"]
    x0, y0, x1, y1 = box
    pen_px, gain = measure_pen(ctx["original"], ctx["gray"], ctx["bg"], ctx["strong"], box, ctx["blur_sigma"])
    core = ink_colour(ctx["original"], ctx["strong"], box)
    op = ctx["orig_paper"][y0:y1, x0:x1].reshape(-1, 3).astype(np.float32).mean(axis=0)
    paper = ctx["page_paper"][y0:y1, x0:x1].reshape(-1, 3).astype(np.float32).mean(axis=0)
    # The pen darkens paper by (op - core) as seen through the blur; the real
    # graphite is darker by `gain`. Apply that darkening to the paper here now.
    darken = (op - core) * gain * style["dark"] * paper / np.maximum(op, 1)
    ink = np.clip(paper - darken, 0, 255)

    # Start the new answer where the old ink started, so spacing after "=" is kept.
    left = min(st[:, 0].min() for st in strokes)
    strokes = [st - [left, 0] for st in strokes]

    # Size: one unit of the drawing is 1/100th of the old answer's height. If the
    # new answer is longer and would run into whatever is written next, write it
    # smaller - the same thing a person does when they are short of room.
    unit = (y1 - y0) / 100
    wide = max(st[:, 0].max() for st in strokes)
    room = room_to_write(ctx, box)
    if wide * unit > room:
        unit = max(room / wide, unit * MIN_SHRINK)
    pen = pen_px * style["width"] * min(1.0, (unit / ((y1 - y0) / 100)) ** 0.5)

    cov, ox, oy = render(strokes, unit, pen, rng, style)
    top = y1 - int(round(100 * unit))          # sits on the old answer's baseline
    blend(page, cov, (x0 - ox, top - oy), ink, paper, ctx["blur_sigma"] * style["blur"])
    return pen_px, gain


def main():
    # Answers can contain characters the Windows console cannot show.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if a != "--redraw"]
    redraw = "--redraw" in sys.argv
    if len(args) != 1:
        sys.exit("Usage: python write.py <photo> [--redraw]")
    load_dotenv()
    ctx = load_page(Path(args[0]))
    src, out_dir, graded = ctx["src"], ctx["out_dir"], ctx["graded"]
    page = ctx["erased"].copy()
    rng = np.random.default_rng(1)

    strips = []
    for p in graded["to_fix"]:
        if "ink_box" not in p:
            continue
        pen_px, gain = write_answer(ctx, page, p, get_strokes(ctx, p, redraw), rng)
        strips.append(compare_strip(ctx["original"], page, p["ink_box"]))
        print(f"  pen width {pen_px:.2f}px, darkness x{gain:.2f}, blur {ctx['blur_sigma']:.2f}")
    (out_dir / f"{src.stem}-graded.json").write_text(json.dumps(graded, indent=2))

    cv2.imwrite(str(out_dir / f"{src.stem}-corrected.png"), page)
    if strips:
        w = max(s.shape[1] for s in strips)
        strips = [np.pad(s, ((0, 8), (0, w - s.shape[1]), (0, 0)), constant_values=255) for s in strips]
        cv2.imwrite(str(out_dir / f"{src.stem}-compare.png"), np.vstack(strips))
    print(f"Saved out/{src.stem}-corrected.png and out/{src.stem}-compare.png")


if __name__ == "__main__":
    main()
