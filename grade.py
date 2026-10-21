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
- `correct_answer` is the text that will actually be written onto the paper in this
  person's handwriting, in the gap the wrong answer leaves behind. So give only what
  belongs on the page: their notation, no words, no explanation in brackets, no
  working out - and about as many characters as the wrong answer it replaces. If a
  digit repeats forever, write it the short way this person writes it ("..." or a
  bar), not in words.
- if it is plain arithmetic or algebra, give `check_expression`: the problem as a
  Python/SymPy expression whose value is the correct answer (e.g. "2+2", "Rational(3,4)*8").
  Otherwise use an empty string.
- if the answer is wrong because the working above it is wrong, list each wrong line
  of working in `work_fixes`. For each: `written` is that whole line exactly as it
  appears, `correct` is the whole line rewritten so it is right, and `box` is a tight
  box around that entire line of working - the whole line, not just the part that
  changes, and nothing from the lines above or below it. Be careful with these boxes:
  the line gets rubbed out and rewritten, so a box in the wrong place rubs out the
  wrong thing. Same rules as the answer: `correct` is written onto the page in this
  person's hand, so keep the same notation and roughly the same length as the line it
  replaces. Leave `work_fixes` empty if the working is fine, or if you cannot pin down
  which lines went wrong.

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
                    "work_fixes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "written": {"type": "string"},
                                "correct": {"type": "string"},
                                "box": {
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
                            "required": ["written", "correct", "box"],
                            "additionalProperties": False,
                        },
                    },
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
                    "is_correct", "check_expression", "work_fixes", "answer_box",
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


def fixes_for(p):
    """Everything to rewrite for one problem: the answer first, then the working.

    Each one looks the same to the later steps, so erase.py and write.py do not
    care whether they are fixing a final answer or a line of working.
    """
    if p["is_correct"] or p["verified"] is False:
        return []
    fixes = [{"problem": p["problem"], "kind": "answer",
              "written_answer": p["written_answer"],
              "correct_answer": p["correct_answer"],
              "answer_box": p["answer_box"]}]
    for w in p.get("work_fixes", []):
        if w["written"].strip() and w["correct"].strip() != w["written"].strip():
            fixes.append({"problem": p["problem"], "kind": "work",
                          "written_answer": w["written"],
                          "correct_answer": w["correct"],
                          "answer_box": w["box"]})
    return fixes


def draw_boxes(img, problems, out_path):
    W, H = img.size
    d = ImageDraw.Draw(img)
    for p in problems:
        b = p["answer_box"]
        box = (b["x0"] * W / 1000, b["y0"] * H / 1000, b["x1"] * W / 1000, b["y1"] * H / 1000)
        colour = (0, 170, 0) if p["is_correct"] else (220, 0, 0)
        d.rectangle(box, outline=colour, width=max(3, W // 300))
        for f in fixes_for(p)[1:]:
            b = f["answer_box"]
            d.rectangle((b["x0"] * W / 1000, b["y0"] * H / 1000,
                         b["x1"] * W / 1000, b["y1"] * H / 1000),
                        outline=(240, 140, 0), width=max(2, W // 400))
    img.save(out_path)


def main():
    # Answers can contain characters the Windows console cannot show.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
        for w in fixes_for(p)[1:]:
            print(f"WORK   {w['written_answer']}  ->  {w['correct_answer']}")
    if not problems:
        print("No problems found on the page.")

    to_fix = [f for p in problems for f in fixes_for(p)]
    (out_dir / f"{src.stem}-graded.json").write_text(json.dumps(
        {"image": str(src), "problems": problems, "to_fix": to_fix}, indent=2))
    draw_boxes(img, problems, out_dir / f"{src.stem}-graded.png")
    answers = sum(1 for f in to_fix if f["kind"] == "answer")
    counts = f"{answers} answer(s) to fix, {len(to_fix) - answers} line(s) of working."
    print("\n" + counts + " Picture with boxes: " + f"out/{src.stem}-graded.png")


if __name__ == "__main__":
    main()
