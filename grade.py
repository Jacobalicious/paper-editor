"""Step 1: read a photo of handwritten work, find the wrong answers.

Usage:  python grade.py samples/example-2plus2.png

Prints what it found and saves out/<name>-graded.png with the answer boxes drawn
on (green = right, red = wrong), plus out/<name>-graded.json for later steps.
"""

import base64
import io
import json
import sys
from pathlib import Path

import anthropic
import sympy
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps

MODEL = "claude-opus-5-5"
MAX_SIDE = 1568  # longest edge sent to Claude; larger gets downscaled anyway

PROMPT = """This is a photo of handwritten work (maths or similar).

Find every problem on the page and its final written answer. For each one:
- transcribe the problem and the written answer exactly as written
- work out the correct answer yourself
- say whether the written answer is correct
- give a tight box around ONLY the written final answer (not the question, not the "="),
  as x0, y0, x1, y1 on a 0-1000 scale of the image width and height
- if it is plain arithmetic or algebra, give `check_expression`: the problem as a
  Python/SymPy expression whose value is the correct answer (e.g. "2+2", "Rational(3,4)*8").
  Otherwise use an empty string.

Ignore faint writing that shows through from the other side of the page.
Only report answers you can actually read."""

SCHEMA = {
    "type": "object",
    "properties": {
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "written_answer": {"type": "string"},
                    "correct_answer": {"type": "string"},
                    "is_correct": {"type": "boolean"},
                    "check_expression": {"type": "string"},
                    "answer_box": {
                        "type": "object",
                        "properties": {
                            "x0": {"type": "number"},
                            "y0": {"type": "number"},
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                        },
                        "required": ["x0", "y0", "x1", "y1"],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "problem", "written_answer", "correct_answer",
                    "is_correct", "check_expression", "answer_box",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["problems"],
    "additionalProperties": False,
}


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return img


def to_base64_jpeg(img):
    small = img.copy()
    small.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def ask_claude(img):
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": to_base64_jpeg(img),
                }},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to read this image.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["problems"]


def same_value(a, b):
    try:
        return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except (sympy.SympifyError, TypeError, SyntaxError):
        return None


def double_check(p):
    """Calculator check. Sets p['verified'] to True/False/None (None = can't check)."""
    expr = p["check_expression"].strip()
    if not expr:
        p["verified"] = None
        return
    try:
        truth = sympy.sympify(expr)
    except (sympy.SympifyError, TypeError, SyntaxError):
        p["verified"] = None
        return
    claude_agrees = same_value(truth, p["correct_answer"])
    written_ok = same_value(truth, p["written_answer"])
    p["verified"] = bool(claude_agrees) and (written_ok is None or written_ok == p["is_correct"])


def draw_boxes(img, problems, out_path):
    W, H = img.size
    d = ImageDraw.Draw(img)
    for p in problems:
        b = p["answer_box"]
        box = (b["x0"] * W / 1000, b["y0"] * H / 1000, b["x1"] * W / 1000, b["y1"] * H / 1000)
        colour = (0, 170, 0) if p["is_correct"] else (220, 0, 0)
        d.rectangle(box, outline=colour, width=max(3, W // 300))
    img.save(out_path)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python grade.py <photo>")
    load_dotenv()
    src = Path(sys.argv[1])
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    img = load_image(src)
    problems = ask_claude(img)
    for p in problems:
        double_check(p)

    for p in problems:
        mark = "OK   " if p["is_correct"] else "WRONG"
        line = f"{mark}  {p['problem']}  written: {p['written_answer']}"
        if not p["is_correct"]:
            line += f"  ->  should be {p['correct_answer']}"
        if p["verified"] is False:
            line += "   (calculator disagrees - will NOT change this one)"
        print(line)
    if not problems:
        print("No problems found on the page.")

    to_fix = [p for p in problems if not p["is_correct"] and p["verified"] is not False]
    (out_dir / f"{src.stem}-graded.json").write_text(json.dumps(
        {"image": str(src), "problems": problems, "to_fix": to_fix}, indent=2))
    draw_boxes(img, problems, out_dir / f"{src.stem}-graded.png")
    print(f"\n{len(to_fix)} answer(s) to fix. Picture with boxes: out/{src.stem}-graded.png")


if __name__ == "__main__":
    main()
