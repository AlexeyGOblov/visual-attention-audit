# -*- coding: utf-8 -*-
"""Predicted attention map: where a model trained on gaze data expects the eye to go.

This is the one part of the toolkit that uses a neural network, and it is optional on
purpose. The rest of the project runs on numpy, scipy and Pillow and ships no weights.

Model: MSI-Net (Kroner, Senden, Driessens, Goebel, Neural Networks 2020), the SALICON
base model published by the author under MIT. Converted once to ONNX, so running it
needs `onnxruntime` and nothing else - no TensorFlow, no GPU, about 190 ms a frame.

    python attention.py frame.png --kind marketplace_card --out result/

What this is and is not, in the terms this project already uses:

  * It predicts a *density of attention*, not fixations and not their order. There is no
    scanpath here and there will not be one: on interfaces two real people agree with
    each other only about half the time, so a drawn trajectory would be a guess wearing
    the costume of a measurement.
  * It works on product frames and it does not work on laid-out screens, and both of
    those are measured rather than assumed. Against 486 e-commerce frames with recorded
    gaze it gains 0.77 bits per fixation over knowing only where people tend to look;
    against 990 interface frames it *loses* 0.25 bits to the same kind of baseline. So
    it is refused on `ui_screen`, `landing_page`, `results_grid`, `slide` and `banner_ad`
    unless `--anyway` is passed. See reference/validation.md for both runs.
  * Natural scenes are looked at from the centre outwards; interfaces are looked at from
    the top left, so a photographic model overstates the middle of a screen. On product
    imagery that tilt does not appear: measured across 486 frames with recorded gaze, the
    gaze itself is centred, and the map gains 0.77 bits per fixation over that centring.
    `centre_bias_r` below says how much of any given map is just position, so the number
    is visible instead of being silently believed.
  * It resizes every frame to about 320 px before looking at it. Lettering smaller than
    that survives nowhere in this map. That question belongs to `text_mass` in clutter.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinds  # noqa: E402

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "msi-net-salicon.onnx",
)

# Whether this map has been put in front of recorded gaze for this kind of frame, and
# what happened. `ok` is False where the acceptance test found the map worse than simply
# knowing where people tend to look. See reference/validation.md for both runs.
STANDING = {
    "photo": (True, "the material the model was trained on",
              "SALICON, natural photographs"),
    "marketplace_card": (True,
                         "PASSED on product imagery: +0.77 bits per fixation over that "
                         "data's own centre bias, shuffled AUC 0.76 against 0.50. It "
                         "still never sees the small lettering, because the frame is "
                         "shrunk to about 320 px before the model looks",
                         "SalECI, 486 frames with recorded gaze"),
    "packaging": (True,
                  "product imagery, the same class of frame the map passed on",
                  "SalECI by proxy"),
    "banner_ad": (False,
                  "FAILED on designed layouts: on posters the map is 0.23 bits per "
                  "fixation WORSE than simply knowing where people look at posters",
                  "UEyes posters, recorded gaze"),
    "slide": (False,
              "FAILED on designed layouts: on posters, the nearest thing to a slide in "
              "the data, the map is 0.23 bits per fixation worse than the centre bias",
              "UEyes posters, recorded gaze"),
    "ui_screen": (False,
                  "FAILED on interfaces: 0.07 to 0.42 bits per fixation WORSE than "
                  "knowing where people look at that sort of screen, on 990 frames. "
                  "It is trained on photographs and pulls to the centre; interfaces are "
                  "read from the top left",
                  "UEyes desktop, mobile and web, recorded gaze"),
    "landing_page": (False,
                     "FAILED: on web pages the map is 0.29 bits per fixation worse "
                     "than the centre bias of web pages",
                     "UEyes web, recorded gaze"),
    "results_grid": (False,
                     "FAILED: a page of competing tiles is a laid-out screen, and on "
                     "every laid-out screen measured the map lost to the centre bias",
                     "UEyes, recorded gaze"),
}


# ------------------------------------------------------------------ the model

def _target_shape(h: int, w: int) -> tuple[int, int]:
    """The three input shapes the published model accepts, picked by aspect ratio."""
    ratio = h / w
    options = {(320, 320): abs(ratio - 1.0),
               (240, 320): abs(ratio - 240 / 320),
               (320, 240): abs(ratio - 320 / 240)}
    return min(options, key=options.get)


def predict(path: str, model: str = DEFAULT_MODEL) -> tuple[np.ndarray, np.ndarray]:
    """Return the attention map at the frame's own size, and the frame itself."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit(
            "the attention map needs onnxruntime: pip install -r "
            "requirements-attention.txt\nThe clutter measures do not need it."
        ) from None
    if not os.path.exists(model):
        raise SystemExit(
            f"no model at {model}.\nBuild it once with scripts/get_weights.py, which "
            f"downloads the MIT-licensed weights and converts them to ONNX."
        )

    im = Image.open(path).convert("RGB")
    w, h = im.size
    th, tw = _target_shape(h, w)
    scale = min(th / h, tw / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32)

    vpad, hpad = th - nh, tw - nw
    top, left = vpad // 2, hpad // 2
    batch = np.pad(small[None], ((0, 0), (top, vpad - top), (left, hpad - left), (0, 0)))

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: batch.astype(np.float32)})[0]
    out = out[0, top:out.shape[1] - (vpad - top), left:out.shape[2] - (hpad - left), 0]

    full = np.asarray(Image.fromarray(out).resize((w, h), Image.BILINEAR), dtype=np.float64)
    full = np.clip(full - full.min(), 0, None)
    total = full.sum()
    return (full / total if total > 0 else full), np.asarray(im)


# ------------------------------------------------------------------ reading it

def centre_bias(h: int, w: int) -> np.ndarray:
    """A blob in the middle that has not looked at the image. The floor to beat."""
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    blob = np.exp(-(y ** 2 + x ** 2) / (2 * 0.45 ** 2))
    return blob / blob.sum()


def read_map(att: np.ndarray) -> dict:
    h, w = att.shape
    flat = np.sort(att.ravel())[::-1]
    top_tenth = float(flat[:max(1, len(flat) // 10)].sum())
    base = centre_bias(h, w)
    r = float(np.corrcoef(att.ravel(), base.ravel())[0, 1])

    quadrants = {
        "top_left": float(att[:h // 2, :w // 2].sum()),
        "top_right": float(att[:h // 2, w // 2:].sum()),
        "bottom_left": float(att[h // 2:, :w // 2].sum()),
        "bottom_right": float(att[h // 2:, w // 2:].sum()),
    }
    peak = np.unravel_index(att.argmax(), att.shape)
    return {
        "concentration": round(top_tenth, 4),
        "centre_bias_r": round(r, 4),
        "peak": [round(float(peak[1]) / w, 3), round(float(peak[0]) / h, 3)],
        "quadrants": {k: round(v, 4) for k, v in quadrants.items()},
    }


def overlay(frame: np.ndarray, att: np.ndarray, path: str) -> None:
    """The map over the frame, hot where attention is predicted to land."""
    a = att / att.max() if att.max() > 0 else att
    heat = np.stack([np.clip(1.6 * a, 0, 1),
                     np.clip(1.6 * a - 0.5, 0, 1),
                     np.clip(1.6 * a - 1.1, 0, 1)], axis=-1)
    # Cold areas keep the frame as it is. Only the warm ones are painted over, or the
    # whole design goes dark and the reader blames the design rather than the overlay.
    weight = np.clip(a * 1.3, 0, 0.8)[..., None]
    blend = weight * heat * 255 + (1 - weight) * frame
    Image.fromarray(blend.astype(np.uint8)).save(path)


# ------------------------------------------------------------------ driver

def selftest(model: str) -> int:
    """One blob on a flat field. The model has to find it, and beat the centre blob."""
    print("1. The model loads and runs")
    field = np.full((320, 320, 3), 128, dtype=np.uint8)
    field[40:100, 220:280] = (255, 40, 40)          # off-centre, high contrast
    tmp = os.path.join(os.path.dirname(model) or ".", "_selftest.png")
    Image.fromarray(field).save(tmp)
    try:
        att, _ = predict(tmp, model)
    finally:
        os.remove(tmp)
    stats = read_map(att)
    print(f"   map sums to {att.sum():.4f}, peak at {stats['peak']}")

    print("2. The peak lands on the blob, not in the middle")
    px, py = stats["peak"]
    on_blob = 0.66 < px < 0.92 and 0.10 < py < 0.36
    print(f"   blob spans x 0.69-0.88, y 0.13-0.31; peak {stats['peak']}: {on_blob}")

    print("3. The map is not just a centre blob")
    print(f"   correlation with the centre bias {stats['centre_bias_r']:.3f}")
    beats_centre = stats["centre_bias_r"] < 0.9

    ok = on_blob and beats_centre and abs(att.sum() - 1.0) < 1e-6
    print(f"\nattention self-test {'passed' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Predicted attention map (MSI-Net, SALICON weights, MIT)",
        epilog=kinds.catalogue(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("images", nargs="*", help="frames to predict")
    ap.add_argument("--kind", choices=kinds.NAMES, metavar="KIND",
                    help="what the frame is; see the list below")
    ap.add_argument("--out", default=None, help="directory for maps and overlays")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to the ONNX model")
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--anyway", action="store_true",
                    help="draw the map even on a kind where it failed acceptance")
    ap.add_argument("--selftest", action="store_true", help="verify the model runs")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.model)
    if not args.images:
        ap.error("pass frames to predict, or --selftest")
    if not args.kind:
        ap.error("--kind is required here too. It does not change the arithmetic, it "
                 "changes how far the model is from anything it was trained on.\n\n"
                 + kinds.catalogue())

    kind = kinds.get(args.kind)
    ok, why, source = STANDING.get(
        kind.name, (False, "no check has been run on this kind", "nothing"))
    print(f"kind: {kind.name} - {kind.summary}")
    print(f"against recorded gaze ({source}): {why}")

    if not ok and not args.anyway:
        raise SystemExit(
            f"\nrefusing to draw a map for {kind.name}.\n\n"
            f"This is not caution, it is a measurement: on this kind of frame the map "
            f"predicts real gaze WORSE than a map that never looked at the image and "
            f"only knows where people tend to look. Presenting it would be presenting "
            f"something less informative than a blur, with the authority of a picture.\n\n"
            f"What to use instead: the clutter measures and text_mass from clutter.py, "
            f"which are not affected, and your own list of what should be noticed first.\n\n"
            f"If you want to look at it anyway, pass --anyway. The output is stamped."
        )
    if not ok:
        print("!! NOT ACCEPTED on this kind: shown because --anyway was passed. "
              "Do not put this in a report.")
    print("this is a predicted density of attention, not a measurement of anyone's gaze")
    print()

    out = []
    for path in args.images:
        att, frame = predict(path, args.model)
        stats = read_map(att)
        stats["file"] = os.path.basename(path)
        stats["kind"] = kind.name

        if args.out:
            os.makedirs(args.out, exist_ok=True)
            stem = os.path.splitext(os.path.basename(path))[0]
            norm = att / att.max() if att.max() > 0 else att
            raw = os.path.join(args.out, f"{stem}-attention.png")
            over = os.path.join(args.out, f"{stem}-attention-over.png")
            Image.fromarray((norm * 255).astype(np.uint8)).save(raw)
            overlay(frame, att, over)
            stats["map"], stats["overlay"] = raw, over

        q = stats["quadrants"]
        print(f"{stats['file']}: peak at {stats['peak']}, top tenth of the frame holds "
              f"{stats['concentration']:.0%} of predicted attention")
        print(f"    quadrants  top-left {q['top_left']:.0%}  top-right {q['top_right']:.0%}"
              f"  bottom-left {q['bottom_left']:.0%}  bottom-right {q['bottom_right']:.0%}")
        print(f"    correlation with a plain centre blob: {stats['centre_bias_r']:.2f}", end="")
        if stats["centre_bias_r"] > 0.8:
            print(" - most of this map is where the frame is, not what is on it")
        else:
            print()
        out.append(stats)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
