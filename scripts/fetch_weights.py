"""Download the depth model once (needed by the video and photo tiers).

    python scripts/fetch_weights.py

Weights are cached by Hugging Face under ~/.cache/huggingface and are NOT stored
in the repo. Needs `pip install torch transformers` (see requirements-models.txt).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.depth_models import DEFAULT_MODEL

from huggingface_hub import snapshot_download

path = snapshot_download(DEFAULT_MODEL)
print("model cached at", path)
