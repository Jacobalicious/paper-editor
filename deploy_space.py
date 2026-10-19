"""Put the phone app on a private Hugging Face Space (runs without this PC).

Usage:  python deploy_space.py
Needs this PC logged in to Hugging Face once (hf auth login).

Copies only the app's code (no photos, no .env), and copies ANTHROPIC_API_KEY and
APP_PASSWORD from .env into the Space's hidden secrets.
"""

import sys
import tempfile
from pathlib import Path
import shutil

import gradio
from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).parent
SPACE = "sushiwushicat/paper-editor"
FILES = ["app.py", "grade.py", "erase.py", "write.py", "favorites.json"]
REQUIREMENTS = """anthropic
opencv-python-headless
numpy
pillow
sympy
python-dotenv
scikit-image
"""
README = f"""---
title: Paper Editor
emoji: ✏️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "3.12"
app_file: app.py
pinned: false
---

Private. Upload a photo of handwritten work; wrong answers come back corrected.
"""


def main():
    env = dotenv_values(ROOT / ".env")
    missing = [k for k in ("ANTHROPIC_API_KEY", "APP_PASSWORD") if not env.get(k)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("This PC isn't logged in to Hugging Face yet. Run: hf auth login")
    print(f"Logged in to Hugging Face as {user}")

    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.add_space_secret(SPACE, "ANTHROPIC_API_KEY", env["ANTHROPIC_API_KEY"])
    api.add_space_secret(SPACE, "APP_PASSWORD", env["APP_PASSWORD"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in FILES:
            shutil.copy(ROOT / f, tmp / f)
        (tmp / "requirements.txt").write_text(REQUIREMENTS)
        (tmp / "README.md").write_text(README, encoding="utf-8")
        api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=tmp,
                          commit_message="Update app")
    print(f"Done. Open https://huggingface.co/spaces/{SPACE}")
    print("First start takes a few minutes while it installs.")


if __name__ == "__main__":
    main()
