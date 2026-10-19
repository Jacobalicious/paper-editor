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


with gr.Blocks(title="Paper Editor") as app:
    gr.Markdown("## Paper editor\nTake a photo of the work. Wrong answers come back fixed.")
    photo = gr.Image(type="filepath", sources=["upload", "webcam"], label="Photo of the work")
    go = gr.Button("Fix it", variant="primary")
    result = gr.Image(label="Corrected (tap and hold to save)", format="png")
    log = gr.Textbox(label="What changed", lines=6)
    go.click(correct, inputs=photo, outputs=[result, log])


def main():
    load_dotenv(ROOT / ".env")
    password = os.getenv("APP_PASSWORD")
    if not password:
        password = secrets.token_urlsafe(9)
        set_key(str(ROOT / ".env"), "APP_PASSWORD", password)
        print("Made a phone password and saved it in .env (APP_PASSWORD).")
    # Port 7860 is already used by Stable Diffusion on this PC.
    app.launch(server_name="0.0.0.0", server_port=7870, auth=("jacob", password),
               share="--share" in sys.argv)


if __name__ == "__main__":
    main()
