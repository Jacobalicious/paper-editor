# paper-editor — plan

Goal: **one step.** Drop in a phone photo of handwritten work → get back the same
photo with every wrong answer replaced by the right one, in the writer's own
handwriting, with no visible edit. No selecting, no typing.

Test image: `samples/example-2plus2.png` — "2 + 2 = 5" should come out "2 + 2 = 4".

## The pipeline

```
photo ─► 1 read & grade ─► 2 locate exactly ─► 3 erase ─► 4 harvest handwriting
      ─► 5 write new answer ─► 6 make it look photographed ─► 7 self-check ─► output
```

### 1. Read and grade (Claude API, vision)
- Send the photo to Claude. It returns JSON: every problem on the page, the
  written answer, whether it's right, the correct answer, and a rough box around
  each answer.
- To make the maths reliable, Claude also works the problem out in steps and, for
  arithmetic/algebra, we double-check with a small calculator (SymPy) before we
  change anything. If Claude and the check disagree, leave that answer alone.
- Only the **final answer** gets replaced in v1. Fixing working-out lines is a
  later stage (it means rewriting several lines, which is far harder to hide).

### 2. Locate exactly
Claude's boxes are approximate. Inside each one, find the actual ink strokes
(adaptive threshold + connected pieces) so we know the exact pixels of the
wrong answer, its height, slant and baseline.

### 3. Erase the old answer
- Mask just the ink strokes (slightly grown), not the whole box.
- Fill with **LaMa** inpainting, which is good at repeating paper texture,
  ruled lines and the faint bleed-through visible in the example. OpenCV
  inpainting as fallback.

### 4. Harvest the writer's handwriting — automatically
No sample sheet. The tool builds a glyph library from the page itself:
- Claude (or OCR) labels each character on the page with a box: "this is a 2",
  "this is a +". We cut those out as real ink samples.
- Needed digit is on the page → use a real copy (best possible match), picking a
  different copy each time so repeats aren't identical.
- Needed digit isn't on the page (like the 4 in the example) → generate it with
  **[One-DM](https://github.com/dailenson/One-DM)**, which imitates a handwriting
  style from a single sample, using cut-outs from the page as the style
  reference. [DiffusionPen](https://github.com/koninik/DiffusionPen) (about 5
  samples) as the alternative.
- Optional upgrade: a glyph library that grows across photos of the same person's
  work, so over time almost every character is real ink.

### 5. Write the new answer
- Size it to match neighbouring characters, sit it on the same baseline, match
  the slant, and space digits the way the writer spaces them.
- Match stroke thickness and ink colour, sampled from the writer's real
  strokes nearby.

### 6. Make it look photographed
- Blend with **multiply** (ink darkens paper, so paper texture and bleed-through
  still show under the strokes) instead of pasting a patch.
- Match blur, grain and the lighting gradient across the page.
- Re-save the whole image at the original JPEG quality so compression is even
  everywhere.

### 7. Self-check, then output
- Send the result back to Claude: "read every answer" (does it now say 4?) and
  "does any part of this look edited?" Retry with different settings if either
  fails.
- Automatic checks: noise and sharpness inside vs outside the edit, plus an
  error-level-analysis image. If the edit stands out, retry.
- Output the corrected photo, plus a small report of what was changed.

## How it's used
- v1: drop a photo into an `in/` folder, run one command, get the result in `out/`.
- v2: a small web page: upload photo → download corrected photo.

## Build order (each step is testable on its own)
1. Grading: photo → Claude → list of wrong answers with correct ones and boxes.
   Test: does it catch the 5 and say 4?
2. Locate + erase. Test: the 5 vanishes and the patch is invisible.
3. Same-page glyph copy + writing + blending. Test with an answer whose digits
   are already on the page.
4. One-DM for missing characters (the 4 in the example). This is the hardest and
   heaviest step (needs PyTorch, ideally a graphics card).
5. Self-check loop.
6. Web page.

## What's needed from Jacob, and when
- **Claude API key** — at build step 1. Don't paste it into chat; it goes in a
  private `.env` file in this folder, which is kept off GitHub.
- **More test photos** — at step 3: a few real pages with mistakes, ideally
  different pens and lighting.

## Decision log
- 2026-10-19: One-DM shelved for now. This PC's GTX 750 Ti is too old for current
  PyTorch, and One-DM targets Linux + multi-GPU + Python 3.8. Instead Claude sketches
  the new answer as pen strokes after studying the writer's line, and write.py renders
  them with the measured pen width, darkness, grain and blur. First result on the
  example is convincing at normal size.

## Tools
Python, OpenCV, NumPy, Pillow, SymPy, the Anthropic Python SDK;
`simple-lama-inpainting` for LaMa; PyTorch + One-DM at step 4.
