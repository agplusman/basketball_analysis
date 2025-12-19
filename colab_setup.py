"""
Utilities to bootstrap this project inside Google Colab (or other transient GPU hosts).

Examples (Colab cells):
```
!python colab_setup.py --install-deps --download-models --models-dir models
!python main.py input_videos/video1.mp4 --output_video output_videos/output_result.avi
```
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, Iterable


MODEL_FILES: Dict[str, str] = {
    "player_detector.pt": "1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy",
    "ball_detector_model.pt": "1KejdrcEnto2AKjdgdo1U1syr5gODp6EL",
    "court_keypoint_detector.pt": "1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0",
}


def ensure_package(package: str) -> None:
    """Install a single package with pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_gdown() -> None:
    """Install gdown if it is missing."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        ensure_package("gdown")


def install_requirements(requirements_path: str) -> None:
    """Install requirements (useful for Colab or RunPod bootstrap)."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", requirements_path]
    )


def download_models(models_dir: str) -> None:
    """Download pretrained weights into the target directory using gdown."""
    ensure_gdown()
    import gdown

    os.makedirs(models_dir, exist_ok=True)
    for filename, file_id in MODEL_FILES.items():
        output_path = os.path.join(models_dir, filename)
        if os.path.exists(output_path):
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Colab/RunPod bootstrap helper")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install packages from requirements.txt",
    )
    parser.add_argument(
        "--requirements-path",
        default="requirements.txt",
        help="Path to requirements file",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download pretrained weights into --models-dir",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store downloaded model weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.install_deps:
        install_requirements(args.requirements_path)

    if args.download_models:
        download_models(args.models_dir)


if __name__ == "__main__":
    main()
