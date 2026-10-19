# paper-editor — plan

Goal: take a phone photo of handwritten work, pick a wrong answer, and replace it
with the right one in the writer's own handwriting, so the edit is invisible.

Test image: `samples/example-2plus2.png` — "2 + 2 = 5", fix the 5 into a 4.

## What makes this hard (from the example photo)

1. **Bleed-through.** The page is thin; faint math from the back/next page shows
   through right under the "5". Erasing the 5 to flat white would leave an obvious
   clean patch. The fill has to recreate that ghost texture.
2. **Lighting gradient.** The paper goes from bright at the top to grey at the bottom
   and bluish at the edges. The new ink and the filled paper must follow it.
3. **Pen character.** Thin, slightly grey pencil/fine pen, with uneven pressure and a
   slightly separate top bar on the 5. A clean font-like "4" would stand out.
4. **Missing glyph.** There is no "4" anywhere in the photo to copy. So we either
   generate one in this style, or get a sample of the writer's "4".
5. **Camera artifacts.** Blur, noise, JPEG compression and slight perspective. Pasted
   content that is sharper or cleaner than the rest is the #1 giveaway.

## Approach — three stages

### Stage 1: Erase the old answer
- User drags a box around the wrong answer.
- Build an ink mask inside the box (adaptive threshold → only the dark strokes, a
  little dilated), so we erase strokes, not the whole rectangle.
- Fill with **LaMa** inpainting (good at repeating textures like paper and lines),
  falling back to OpenCV inpainting if LaMa isn't installed.
- Check: the erased patch should have the same brightness, noise and bleed-through
  as its surroundings.

### Stage 2: Get the new characters in the right handwriting
In order of quality, best first:
1. **Glyph library from the writer (most realistic).** Once, the writer fills out a
   sample sheet (0–9, + − × ÷ = and the alphabet, a few of each) with the same pen
   on the same kind of paper. Photograph it; the tool cuts out each character.
   Real ink from the real hand is very hard to tell apart. Pick a random variant
   each time so repeated digits aren't identical copies.
2. **Copy from the same page.** If the needed digit already appears on the page,
   cut it from there (perfect pen, lighting and blur match).
3. **AI handwriting imitation** when neither exists:
   [One-DM](https://github.com/dailenson/One-DM) (works from a single sample) or
   [DiffusionPen](https://github.com/koninik/DiffusionPen) (about 5 samples).
   Outputs a clean black-on-white glyph we then "age" in stage 3.
4. **Whole-region AI text editing** ([DiffUTE](https://arxiv.org/abs/2305.10825),
   AnyText) as a later experiment. Research says these still struggle to keep
   handwriting style consistent, so not the first choice.

### Stage 3: Make it look photographed
- Scale the glyph to match neighbouring characters (height of the "2"s) and line it up on the same baseline.
- Match stroke thickness (thin/thicken with morphology) and ink colour (sample the
  real strokes nearby).
- Blend with **multiply** (ink darkens paper, so bleed-through still shows under it)
  instead of pasting a white-backed patch.
- Match blur, add matching grain, apply the same lighting gradient, and finally
  re-save the whole image at the original JPEG quality so compression is uniform.

### Stage 4: Self-check
- Side-by-side and zoomed diff of before/after.
- Automatic checks: noise level and sharpness inside vs outside the edit, and an
  error-level-analysis (ELA) image. If the edit "glows" in ELA, keep tuning.

## Build order (sketch mode — small steps)
1. `edit.py`: open image, drag a box, erase with OpenCV inpaint. See how bad it looks.
2. Swap in LaMa. Compare.
3. Draw a 4 from a hand-made glyph PNG with multiply blend + blur/noise matching.
4. Sample-sheet cutter → glyph library.
5. Try One-DM for the no-sample case.
6. Small web page (drag box, type answer, download) if it's worth keeping.

## Tools
Python, OpenCV, NumPy, Pillow; `lama-cleaner` / `simple-lama-inpainting` for LaMa;
PyTorch only once we reach the AI-handwriting step (it's a large install).
