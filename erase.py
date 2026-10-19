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

PAD = 0.35      # grow Claude's box by this fraction on each side before searching
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
    """Keep only ink blobs that mostly sit inside Claude's box."""
    x0, y0, x1, y1 = box
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4:
            continue
        ix0, iy0 = max(x, x0), max(y, y0)
        ix1, iy1 = min(x + w, x1), min(y + h, y1)
        overlap = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if overlap >= 0.5 * w * h:
            keep[labels == i] = 255
    return keep


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
        px, py = int((x1 - x0) * PAD), int((y1 - y0) * PAD)
        search = (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))
        strokes = find_answer_ink(mask, search)
        ys, xs = np.nonzero(strokes)
        if len(xs) == 0:
            print(f"Couldn't find ink for '{p['written_answer']}', skipping.")
            continue
        p["ink_bgr"] = [int(v) for v in np.median(img[strokes > 0], axis=0)]
        ink_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
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
