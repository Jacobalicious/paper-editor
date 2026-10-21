"""Step 0: find the sheet of paper in the photo and flatten it.

Usage:  python page.py <photo> [out.png]

A phone photo taken over a desk has the page at an angle, surrounded by whatever
else is in the frame. Everything after this has to point at exact places on the
page, and that is much harder when half the picture is a lap and the page is a
tilted quadrilateral. So: find the paper, straighten it, throw the rest away.

If no page can be found - a close-up with no visible edges, say - the photo is
left exactly as it is.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from erase import ink_mask, paper_background

KEEP = 0.97       # a crop must leave at least this much of the writing on the page
MIN_AREA = 0.20   # a shape smaller than this much of the frame is not the page
MAX_AREA = 0.99   # ...and one this big is just the frame itself


def page_corners(img):
    """The four corners of the sheet of paper, or None.

    The paper is the big bright thing. Its own edges are not reliable - a corner
    can be in shadow, or run off the side of the photo - so rather than hunting
    for a four-sided outline, take the bright region, wrap it in its convex hull
    and use the smallest rectangle that holds it. That survives a dark corner and
    comes out straight even when the page is at an angle.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 2)

    mask = (gray > np.percentile(gray, 55)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    rect = cv2.minAreaRect(cv2.convexHull(max(contours, key=cv2.contourArea)))
    corners = cv2.boxPoints(rect)
    area = cv2.contourArea(corners)
    if not MIN_AREA * gray.size < area < MAX_AREA * gray.size:
        return None
    return order_corners(corners * 4)


def order_corners(pts):
    """Top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def keeps_the_writing(img, corners):
    """Would this crop cut any writing off? A page running past the edge of the
    photo can be mistaken for a smaller one, and losing half a sum is far worse
    than leaving the photo as it came."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = paper_background(gray)
    ink, _ = ink_mask(gray, bg)
    ink = ink & ((bg >= np.percentile(bg, 55)) * 255).astype(np.uint8)  # on paper only
    inside = np.zeros_like(ink)
    middle = corners.mean(axis=0)
    shrunk = (middle + (corners - middle) * 0.98).astype(np.int32)
    cv2.fillConvexPoly(inside, shrunk, 255)
    total = (ink > 0).sum()
    return total == 0 or ((ink > 0) & (inside > 0)).sum() / total >= KEEP


def flatten(img):
    """Crop to the page and square it up. Returns the photo unchanged if unsure."""
    corners = page_corners(img)
    if corners is None or not keeps_the_writing(img, corners):
        return img
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 200 or h < 200:
        return img
    target = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, target), (w, h))


def main():
    if not 2 <= len(sys.argv) <= 3:
        sys.exit("Usage: python page.py <photo> [out.png]")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) == 3 else src
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    out = flatten(img)
    cv2.imwrite(str(dst), out)
    if out.shape == img.shape:
        print("Using the photo as it is: no clear page edges, or a crop would "
              "have cut off some of the writing.")
    else:
        print(f"Page found: {img.shape[1]}x{img.shape[0]} photo -> {out.shape[1]}x{out.shape[0]} page")


if __name__ == "__main__":
    main()
