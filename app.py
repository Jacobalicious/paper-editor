"""Phone-friendly web app: upload a photo of work, get the corrected photo back.

Usage:  python app.py            (only on this PC's Wi-Fi network)
        python app.py --share    (also a temporary link that works anywhere)

Password-protected with APP_PASSWORD from .env (user name: jacob), so nobody else
can spend the Claude credits.
"""

import os
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv, set_key
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
PY = sys.executable
MAX_SIDE = 2000  # phone photos are huge; this is plenty for handwriting


def run(step, photo):
    r = subprocess.run([PY, step, str(photo)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise gr.Error(f"{step} failed: {(r.stderr or r.stdout).strip().splitlines()[-1]}")
    return r.stdout


def correct(upload):
    if not upload:
        raise gr.Error("Take or choose a photo first.")
    (ROOT / "in").mkdir(exist_ok=True)
    photo = ROOT / "in" / f"photo-{datetime.now():%Y%m%d-%H%M%S}.png"
    img = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(photo)

    report = run("grade.py", photo)
    if "0 answer(s) to fix" in report:
        return str(photo), report.strip() + "\n\nNothing to change."
    run("erase.py", photo)
    run("write.py", photo)
    changes = [l for l in report.splitlines() if l.startswith(("WRONG", "OK"))]
    return str(ROOT / "out" / f"{photo.stem}-corrected.png"), "\n".join(changes)


# Phones open the back camera straight away when a file picker asks for capture="environment".
# Gradio has no option for that, so this sets it on the Scan button's hidden file input
# (Gradio places that input right after the button).
REAR_CAMERA_JS = """
<script>
new MutationObserver(() => {
  document.querySelectorAll('#scan-btn + input[type=file]').forEach(i => {
    if (!i.hasAttribute('capture')) { i.setAttribute('capture', 'environment'); i.accept = 'image/*'; }
  });
}).observe(document.documentElement, {childList: true, subtree: true});
</script>
"""
CSS = """
#scan-btn { min-height: 72px; font-size: 22px; }
"""

with gr.Blocks(title="Paper Editor") as app:
    gr.Markdown("## Paper editor\nScan the work. Wrong answers come back fixed.")
    scan = gr.UploadButton("📷 Scan", file_types=["image"], type="filepath",
                           variant="primary", elem_id="scan-btn")
    pick = gr.UploadButton("Choose from photos", file_types=["image"], type="filepath",
                           variant="secondary", elem_id="pick-btn")
    result = gr.Image(label="Corrected (tap and hold to save)", format="png")
    log = gr.Textbox(label="What changed", lines=4)
    scan.upload(correct, inputs=scan, outputs=[result, log])
    pick.upload(correct, inputs=pick, outputs=[result, log])


def main():
    hosted = bool(os.getenv("RENDER") or os.getenv("SPACE_ID"))  # set by Render / HF Spaces
    load_dotenv(ROOT / ".env")
    password = os.getenv("APP_PASSWORD")
    if not password:
        if hosted:
            sys.exit("APP_PASSWORD is missing; refusing to run without a password.")
        password = secrets.token_urlsafe(9)
        set_key(str(ROOT / ".env"), "APP_PASSWORD", password)
        print("Made a phone password and saved it in .env (APP_PASSWORD).")
    page = {"auth": ("jacob", password), "head": REAR_CAMERA_JS, "css": CSS, "show_error": True}
    if hosted:
        app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), **page)
    else:
        # Port 7860 is already used by Stable Diffusion on this PC.
        app.launch(server_name="0.0.0.0", server_port=7870, share="--share" in sys.argv, **page)


if __name__ == "__main__":
    main()
