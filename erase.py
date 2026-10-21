"""Step 2: erase each wrong answer found by grade.py.

Usage:  python erase.py samples/example-2plus2.png
Needs out/<name>-graded.json from grade.py.

Saves out/<name>-erased.png and out/<name>-erase-compare.png (zoomed before/after).
It also writes each answer's exact ink box, height and baseline into the json
for the next step.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PAD = 0.35      # how far past the ends of the box to look for ink, as a
                # fraction of the line's height
PAD_Y = 0.12    # and how far above and below it
INK_DELTA = 28  # how much darker than the paper a pixel must be to count as ink
EDGE_DELTA = 10  # fainter pixels touching real ink (stroke edges, light starts) also count


def paper_background(gray):
    """Estimate the blank paper: close (fills thin dark strokes), then smooth."""
    k = max(15, (min(gray.shape) // 40) | 1)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(closed, (0, 0), k / 3)


def ink_mask(gray, bg):
    diff = cv2.subtract(bg, gray)
    return (diff > INK_DELTA).astype(np.uint8) * 255, diff


def add_faint_edges(strokes, diff):
    near = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return strokes | (((diff > EDGE_DELTA) & (near > 0)).astype(np.uint8) * 255)


def find_answer_ink(mask, box):
    """Keep only ink blobs that mostly sit inside Claude's box, on one line.

    Returns two masks: everything to rub out, and just the full-size characters
    of the line. The second is what the new writing is sized against, so that a
    subscript hanging below does not make the replacement taller.
    """
    x0, y0, x1, y1 = box
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4:
            continue
        ix0, iy0 = max(x, x0), max(y, y0)
        ix1, iy1 = min(x + w, x1), min(y + h, y1)
        overlap = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if overlap >= 0.5 * w * h:
            blobs.append((i, y, h))

    keep, core = np.zeros_like(mask), np.zeros_like(mask)
    on_row, hanging = one_line(blobs)
    for i in on_row:
        keep[labels == i] = 255
        core[labels == i] = 255
    for i in hanging:
        keep[labels == i] = 255
    return keep, core


def one_line(blobs):
    """Drop blobs that belong to the line above or below.

    The box Claude gives is a guess, and the search area around it is wider still,
    so ink from a neighbouring row can sneak in. Everything later is scaled to the
    height of what we find here, so one stray blob from the next line would make
    the replacement writing much too big. Keep what sits on the same row as the
    typical blob: its middle close to theirs.

    Lines of working can be packed so tightly that a character is nearly as tall
    as the gap between rows, so "close" has to mean a good deal less than a whole
    character height. Small characters get a little more room, so that a subscript
    hanging below the line stays with the line it belongs to.
    """
    if len(blobs) < 3:
        return [i for i, _, _ in blobs], []
    mids = np.array([y + h / 2 for _, y, h in blobs])
    heights = np.array([h for _, _, h in blobs], float)
    tall = heights >= np.median(heights)      # full-size characters, not dots
    row = np.median(mids[tall])
    span = max(np.median(heights[tall]), 1.0)
    line, hanging = [], []
    for (i, y, h), m in zip(blobs, mids):
        if abs(m - row) <= 0.55 * span:
            line.append(i)
        # A subscript is small, starts within the line and hangs below it. The
        # line below is small too, but it starts well clear of this one.
        elif h < 0.7 * span and y < row + 0.6 * span and m - row <= span:
            hanging.append(i)
    return line, hanging


def erase(img, strokes):
    """Fill the strokes with paper. Inpaint, then restore matching grain."""
    grow = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    filled = cv2.inpaint(img, grow, 5, cv2.INPAINT_TELEA)

    # Inpainting comes out too smooth; borrow the noise level of the nearby paper.
    ring = cv2.dilate(grow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) & ~grow
    smooth = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32)
    noise_sd = (img.astype(np.float32) - smooth)[ring > 0].std(axis=0)
    rng = np.random.default_rng(0)
    grain = rng.normal(0, 1, img.shape).astype(np.float32) * noise_sd
    grain = cv2.GaussianBlur(grain, (0, 0), 0.6)
    out = filled.astype(np.float32)
    soft = cv2.GaussianBlur(grow.astype(np.float32) / 255, (0, 0), 1.5)[..., None]
    out = out + grain * soft
    return np.clip(out, 0, 255).astype(np.uint8)


def compare_strip(before, after, box, scale=4):
    x0, y0, x1, y1 = box
    pad = (x1 - x0) // 2
    sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
    a, b = before[sl], after[sl]
    a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
    return np.hstack([a, gap, b])


def main():
    # Answers can contain characters the Windows console cannot show.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) != 2:
        sys.exit("Usage: python erase.py <photo>")
    src = Path(sys.argv[1])
    out_dir = Path("out")
    graded_path = out_dir / f"{src.stem}-graded.json"
    graded = json.loads(graded_path.read_text())

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask, diff = ink_mask(gray, paper_background(gray))

    result = img.copy()
    strips = []
    for p in graded["to_fix"]:
        b = p["answer_box"]
        x0, y0, x1, y1 = (int(b["x0"] * W / 1000), int(b["y0"] * H / 1000),
                          int(b["x1"] * W / 1000), int(b["y1"] * H / 1000))
        # The box could be out by about a character either way, so search that much
        # further - in both directions, never by a share of a long line's width.
        # Look wider than the box sideways, where the ink can run past its ends,
        # but barely at all up and down: the line above is only millimetres away
        # and must not be dragged in.
        h = max(y1 - y0, 8)
        px, py = int(h * PAD), int(h * PAD_Y)
        search = (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))
        strokes, core = find_answer_ink(mask, search)
        ys, xs = np.nonzero(core if core.any() else strokes)
        if len(xs) == 0:
            print(f"SKIP   couldn't find '{p['written_answer']}' on the page - left as it was")
            continue
        ink_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        # If what we found is nothing like the size of what was asked for, the box
        # is probably pointing at the wrong thing. Rubbing out the wrong thing is
        # worse than leaving it, so leave it.
        if not 0.4 <= (ink_box[3] - ink_box[1]) / max(y1 - y0, 1) <= 2.0:
            print(f"SKIP   couldn't pin down '{p['written_answer']}' - left as it was")
            continue
        p["ink_bgr"] = [int(v) for v in np.median(img[strokes > 0], axis=0)]
        p["ink_box"] = ink_box
        p["ink_height"] = ink_box[3] - ink_box[1]
        p["baseline_y"] = ink_box[3]
        result = erase(result, add_faint_edges(strokes, diff))
        strips.append(compare_strip(img, result, ink_box))
        print(f"Erased '{p['written_answer']}' at {ink_box}")

    cv2.imwrite(str(out_dir / f"{src.stem}-erased.png"), result)
    if strips:
        w = max(s.shape[1] for s in strips)
        strips = [np.pad(s, ((0, 8), (0, w - s.shape[1]), (0, 0)), constant_values=255) for s in strips]
        cv2.imwrite(str(out_dir / f"{src.stem}-erase-compare.png"), np.vstack(strips))
    graded_path.write_text(json.dumps(graded, indent=2))
    print(f"Saved out/{src.stem}-erased.png and out/{src.stem}-erase-compare.png")


if __name__ == "__main__":
    main()
