# -*- coding: utf-8 -*-
"""Fetch the MSI-Net weights once and convert them to ONNX.

Run this once. Afterwards `attention.py` needs only `onnxruntime`, and the tool never
touches the network again.

    python scripts/get_weights.py

Where the weights come from and why this one:

    https://huggingface.co/alexanderkroner/MSI-Net   (license: mit)

MSI-Net is published by its author with the code under MIT and the model card on
HuggingFace declares MIT for the weights themselves, which is rare in this field: most
saliency models with comparable accuracy are either unlicensed (DeepGaze, where the MIT
line in setup.py is commented out), non-commercial by their own text (UMSI, Imp1k), or
built on a research-only base model (SeekUI, on Qwen2.5-VL-3B). So this is not the most
accurate model available; it is the most accurate one that can be used without asking
anybody's permission.

It is the SALICON base model. The author states plainly that SALICON is mouse movement
over natural photographs standing in for gaze. The checkpoint fine-tuned on web pages
(`fiwi`) is not published here, only in the old TensorFlow 1 release.

The conversion is exact, and that is checked rather than assumed: on the author's own
example image the ONNX output matched the TensorFlow output to 7.7e-07 at the largest
single pixel, with a correlation of 1.0. Re-run the check with --verify if you rebuild.

Converting needs TensorFlow and tf2onnx, which are heavy and are deliberately not in
this project's requirements. Install them into a throwaway environment:

    python -m venv .convert && .convert/Scripts/pip install tensorflow-cpu tf2onnx
    .convert/Scripts/python scripts/get_weights.py
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REPO = "https://huggingface.co/alexanderkroner/MSI-Net/resolve/main"
FILES = ("saved_model.pb", "variables/variables.data-00000-of-00001",
         "variables/variables.index", "example.jpg", "README.md")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED = os.path.join(ROOT, "weights", "msi-net-savedmodel")
ONNX = os.path.join(ROOT, "weights", "msi-net-salicon.onnx")


def download() -> None:
    for name in FILES:
        target = os.path.join(SAVED, name.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            print(f"  have {name}")
            continue
        print(f"  fetching {name}")
        urllib.request.urlretrieve(f"{REPO}/{name}", target)


def convert() -> None:
    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        raise SystemExit(
            "converting needs tensorflow-cpu and tf2onnx, which this project does not "
            "depend on. See the recipe at the top of this file; it takes one throwaway "
            "environment and about five minutes, and you never need them again."
        ) from None
    from tf2onnx import convert as _c  # noqa: F401
    os.system(f'"{sys.executable}" -m tf2onnx.convert --saved-model "{SAVED}" '
              f'--output "{ONNX}" --opset 17')


def verify() -> int:
    """Run the author's own example through both and compare, rather than trusting."""
    import numpy as np
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attention import _target_shape

    im = Image.open(os.path.join(SAVED, "example.jpg")).convert("RGB")
    w, h = im.size
    th, tw = _target_shape(h, w)
    s = min(th / h, tw / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    vp, hp = th - nh, tw - nw
    top, left = vp // 2, hp // 2
    x = np.pad(small[None], ((0, 0), (top, vp - top), (left, hp - left), (0, 0)))
    x = x.astype(np.float32)

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    a = sess.run(None, {sess.get_inputs()[0].name: x})[0].astype(np.float64).ravel()

    import tensorflow as tf
    fn = tf.saved_model.load(SAVED).signatures["serving_default"]
    b = list(fn(tf.constant(x)).values())[0].numpy().astype(np.float64).ravel()

    worst = float(np.abs(a - b).max())
    r = float(np.corrcoef(a, b)[0, 1])
    print(f"largest difference at a single pixel: {worst:.3e}")
    print(f"correlation between the two outputs:  {r:.8f}")
    ok = worst < 1e-5 and r > 0.9999
    print("conversion is faithful" if ok else "CONVERSION DOES NOT MATCH")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="fetch and convert the MSI-Net weights")
    ap.add_argument("--verify", action="store_true",
                    help="compare the ONNX against TensorFlow on the author's example")
    args = ap.parse_args()

    if args.verify:
        return verify()

    print(f"weights from {REPO}  (license: mit)")
    download()
    if os.path.exists(ONNX):
        print(f"already converted: {ONNX}")
        return 0
    print("converting to ONNX")
    convert()
    print(f"done: {ONNX}")
    print("check it with: python scripts/attention.py --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
