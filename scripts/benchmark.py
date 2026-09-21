# -*- coding: utf-8 -*-
"""Does the attention map know anything about the design, or only where the frame is?

This is the acceptance test for mode 3. It is not run by users; it is run before anyone
is allowed to present the attention map as a finding.

    python scripts/benchmark.py --data data/saleci --json result.json

The question it answers is narrow and it is the only one worth asking first. A blurred
blob in the middle of a frame, which has never seen the image, scores AUC 0.78 on the
standard benchmark, and the best classical model reaches 0.48 by correlation against that
blob's 0.45. So "our map correlates with human gaze at 0.6" means nothing on its own. The
number that means something is the margin over a baseline that knows only where people
tend to look regardless of content.

Two baselines are used, and the second is the honest one:

  * a generic centre Gaussian, the same one `attention.py` reports against;
  * the **empirical centre bias of this dataset**, built by averaging the fixation maps of
    one half of it and evaluated on the other half. This is what the literature uses, and
    it is a much harder baseline than the Gaussian, because it has learned the real shape
    of where people look at this kind of image.

Metrics, and why these:

  * **IG** (information gain per fixation, bits) against each baseline. This is the
    headline: it is defined as a gain over a baseline rather than an absolute score, so it
    cannot be inflated by the thing we are worried about.
  * **sAUC** (shuffled AUC), where the negatives are drawn from fixations on *other*
    images. Centre bias helps positives and negatives equally, so it cancels. A map that is
    only centre bias scores 0.5 here.
  * AUC-Judd, NSS, CC, SIM, KLD for comparison with published tables. Read them, but do
    not decide on them.

Data: the SalECI / E-commercial dataset (Li and others, CVPR 2022), MIT licensed,
https://github.com/leafy-lee/E-commercial-dataset - 972 images of e-commerce product
imagery with recorded eye tracking. The only public gaze dataset of this material.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import DEFAULT_MODEL, centre_bias  # noqa: E402

EPS = 1e-12


# ------------------------------------------------------------------ metrics

def _prob(m: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """A probability map with a uniform floor, so that a log is always defined."""
    m = np.clip(m.astype(np.float64), 0, None)
    s = m.sum()
    m = m / s if s > 0 else np.full_like(m, 1.0 / m.size)
    return (1 - floor) * m + floor / m.size


def nss(pred: np.ndarray, fix: np.ndarray) -> float:
    p = pred.astype(np.float64)
    sd = p.std()
    if sd <= 0:
        return 0.0
    z = (p - p.mean()) / sd
    return float(z[fix > 0].mean())


def auc(pred: np.ndarray, fix: np.ndarray) -> float:
    """Area under the ROC, positives at fixations, negatives everywhere else.

    Computed as the Mann-Whitney statistic, which is the exact AUC and handles ties
    properly, instead of sweeping thresholds.
    """
    p = pred.ravel().astype(np.float64)
    hit = fix.ravel() > 0
    n_pos, n_neg = int(hit.sum()), int((~hit).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(p)
    return float((ranks[hit].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def shuffled_auc(pred: np.ndarray, fix: np.ndarray, other: np.ndarray) -> float:
    """AUC with the negatives taken from fixations on other images.

    This is the metric that centre bias cannot buy: a map that is only a centre blob
    scores the same on both sets, so it lands at 0.5.
    """
    p = pred.astype(np.float64)
    pos = p[fix > 0]
    neg = p[other > 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    both = stats.rankdata(np.concatenate([pos, neg]))
    n_pos = pos.size
    return float((both[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * neg.size))


def cc(pred: np.ndarray, dens: np.ndarray) -> float:
    a, b = pred.astype(np.float64).ravel(), dens.astype(np.float64).ravel()
    if a.std() <= 0 or b.std() <= 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sim(pred: np.ndarray, dens: np.ndarray) -> float:
    a, b = _prob(pred, 0.0), _prob(dens, 0.0)
    return float(np.minimum(a, b).sum())


def kld(pred: np.ndarray, dens: np.ndarray) -> float:
    a, b = _prob(pred), _prob(dens)
    return float((b * np.log(b / a)).sum())


def info_gain(pred: np.ndarray, base: np.ndarray, fix: np.ndarray) -> float:
    """Bits of information the map gives per fixation, over the baseline."""
    p, q = _prob(pred), _prob(base)
    at = fix > 0
    if not at.any():
        return float("nan")
    return float(np.mean(np.log2(p[at]) - np.log2(q[at])))


# ------------------------------------------------------------------ the data

def _pairs_saleci(root: str) -> list[dict]:
    """SalECI: ALLSTIMULI/N.jpg with ALLFIXATIONMAPS/N_fixPts.jpg and N_fixMap.jpg."""
    pairs = []
    for fix in sorted(glob.glob(os.path.join(root, "**", "*_fixPts.jpg"), recursive=True)):
        stem = fix[: -len("_fixPts.jpg")]
        dens = next((g for g in (stem + "_fixMap.jpg", stem + "_.fixMap.jpg")
                     if os.path.exists(g)), None)
        base = os.path.basename(stem)
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            hits = [h for h in glob.glob(os.path.join(root, "**", base + ext), recursive=True)
                    if "fix" not in os.path.basename(h).lower()]
            if hits:
                img = hits[0]
                break
        if img and dens:
            pairs.append({"image": img, "fix": fix, "dens": dens,
                          "category": "product", "split": None})
    return pairs


def _pairs_ueyes(root: str, duration: str) -> list[dict]:
    """UEyes: images/ plus saliency_maps/fixmaps_Xs and heatmaps_Xs, with info.csv.

    info.csv carries the category (webpage, desktop, mobile, poster) and the authors'
    own train/test split, so neither has to be invented here.
    """
    import csv
    import re

    def norm(k: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", (k or "").lower()).strip("_")

    meta = {}
    info = next((os.path.join(root, n) for n in ("image_types.csv", "info.csv")
                 if os.path.exists(os.path.join(root, n))), None)
    if info:
        with open(info, newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
            except csv.Error:
                dialect = csv.excel
            for row in csv.DictReader(fh, dialect=dialect):
                keys = {norm(k): (v or "").strip() for k, v in row.items() if k}
                name = next((v for k, v in keys.items()
                             if "image" in k or k in ("name", "file", "filename")), None)
                if not name:
                    continue
                cat = next((v for k, v in keys.items()
                            if "categ" in k or k in ("type", "ui_type")), "unknown")
                split = next((v for k, v in keys.items()
                              if "train" in k or "split" in k or "test" in k), None)
                meta[os.path.splitext(name)[0]] = {"category": cat.lower(),
                                                   "split": split}

    fixdir = os.path.join(root, "saliency_maps", f"fixmaps_{duration}")
    densdir = os.path.join(root, "saliency_maps", f"heatmaps_{duration}")
    pairs = []
    for fix in sorted(glob.glob(os.path.join(fixdir, "*"))):
        base = os.path.splitext(os.path.basename(fix))[0]
        dens = next((g for g in glob.glob(os.path.join(densdir, base + ".*"))), None)
        img = next((g for g in glob.glob(os.path.join(root, "images", base + ".*"))), None)
        if img and dens:
            m = meta.get(base, {"category": "unknown", "split": None})
            pairs.append({"image": img, "fix": fix, "dens": dens,
                          "category": m["category"], "split": m["split"]})
    return pairs


def find_pairs(root: str, duration: str = "3s") -> list[dict]:
    """Image, fixation points and fixation density, however the set is laid out."""
    if os.path.isdir(os.path.join(root, "saliency_maps", f"fixmaps_{duration}")):
        return _pairs_ueyes(root, duration)
    return _pairs_saleci(root)


def load_gray(path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if shape and im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64)


def load_fix(path: str, threshold: int = 128) -> np.ndarray:
    """The fixation points, thresholded, at their own size. Never resampled.

    The dataset stores these as JPEG, and a JPEG of a sparse map of dots rings: on a
    typical frame 217 pixels are real fixations and about 12 000 more are above zero
    purely from compression. Taking `> 0` would hand the whole test to the codec.
    The real points sit on a plateau: the count is identical for any threshold from 10
    to 200, so 128 is safely inside it.

    Resampling is refused for the same reason. Downscaling a sparse binary map loses
    points or invents them; the prediction is resized to the fixations instead.
    """
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) > threshold


# ------------------------------------------------------------------ the run

NORM = (256, 256)   # the space the centre bias is built in, before being fitted to a frame


def empirical_bias(pairs: list, threshold: int, blur: float = 12.0) -> np.ndarray:
    """Where people look on this kind of image regardless of what is on it.

    Built in normalised coordinates, because the frames differ in size and shape, and
    only from the fitting set, so the frames it is judged against are unseen.
    """
    acc = np.zeros(NORM, dtype=np.float64)
    for p in pairs:
        f = load_fix(p["fix"], threshold).astype(np.float64)
        if f.shape != NORM:
            f = np.asarray(Image.fromarray((f * 255).astype(np.uint8))
                           .resize(NORM[::-1], Image.BILINEAR), dtype=np.float64)
        acc += f / max(f.sum(), 1.0)
    acc = ndimage.gaussian_filter(acc, blur)
    return acc / max(acc.sum(), EPS)


def fit_to(bias: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if bias.shape == shape:
        return bias / max(bias.sum(), EPS)
    out = np.asarray(Image.fromarray(bias / bias.max()).resize(shape[::-1], Image.BILINEAR),
                     dtype=np.float64)
    return out / max(out.sum(), EPS)


def main() -> int:
    ap = argparse.ArgumentParser(description="acceptance test for the attention map")
    ap.add_argument("--data", required=True, help="directory holding the dataset")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="use only N evaluation images")
    ap.add_argument("--fix-threshold", type=int, default=128,
                    help="grey level above which a pixel counts as a fixation")
    ap.add_argument("--duration", default="3s", choices=("1s", "3s", "7s"),
                    help="viewing duration, where the data offers a choice (UEyes)")
    ap.add_argument("--shuffle-pool", type=int, default=10,
                    help="how many other frames' fixations form the shuffled-AUC negatives")
    ap.add_argument("--random-split", action="store_true",
                    help="ignore a split declared in the data and take a random half")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pairs = find_pairs(args.data, args.duration)
    if not pairs:
        raise SystemExit(f"no image / fixation pairs found under {args.data}")
    cats = sorted({p["category"] for p in pairs})
    print(f"{len(pairs)} images with recorded gaze in {args.data}")
    print(f"categories: {', '.join(cats)}")

    declared = [p for p in pairs if p["split"]]
    if declared and not args.random_split:
        fit = [p for p in pairs if (p["split"] or "").lower().startswith("train")]
        test = [p for p in pairs if (p["split"] or "").lower().startswith("test")]
        print("using the authors' own train/test split")
    else:
        rng = np.random.default_rng(7)
        order = rng.permutation(len(pairs))
        half = len(pairs) // 2
        fit = [pairs[i] for i in order[:half]]
        test = [pairs[i] for i in order[half:]]
        print("no split declared in the data: fitting on a random half")
    if not fit or not test:
        raise SystemExit("could not form a fit/test split")
    if args.limit:
        test = test[: args.limit]
    print(f"baseline fitted on {len(fit)}, evaluated on {len(test)}")
    print(f"fixation threshold {args.fix_threshold}, duration {args.duration}")

    print("building the empirical centre bias from the fitting set")
    emp_all = empirical_bias(fit, args.fix_threshold)
    emp_cat = {}
    for c in cats:
        same = [p for p in fit if p["category"] == c]
        emp_cat[c] = empirical_bias(same, args.fix_threshold) if len(same) >= 20 else emp_all

    from attention import predict
    rows = []
    for n, p in enumerate(test, 1):
        img, fix, dens = p["image"], p["fix"], p["dens"]
        f = load_fix(fix, args.fix_threshold)
        if not f.any():
            continue
        shape = f.shape
        att, _ = predict(img, args.model)
        pred = np.asarray(Image.fromarray(att).resize(shape[::-1], Image.BILINEAR),
                          dtype=np.float64)
        d = load_gray(dens, shape)
        emp = fit_to(emp_cat[p["category"]], shape)
        gauss = centre_bias(*shape)
        # Negatives for the shuffled AUC: fixations from several other frames, pooled, so
        # that the null distribution is "where people look on this sort of thing" rather
        # than "where they looked on one particular other frame".
        other = np.zeros(shape, dtype=bool)
        for step in range(1, args.shuffle_pool + 1):
            o = load_fix(test[(n + step * 37) % len(test)]["fix"], args.fix_threshold)
            if o.shape != shape:
                o = np.asarray(Image.fromarray(o.astype(np.uint8) * 255)
                               .resize(shape[::-1], Image.NEAREST)) > 128
            other |= o
        if not other.any():
            continue

        rows.append({
            "file": os.path.basename(img),
            "category": p["category"],
            "model": {
                "ig_vs_gauss": info_gain(pred, gauss, f),
                "ig_vs_empirical": info_gain(pred, emp, f),
                "sauc": shuffled_auc(pred, f, other),
                "auc": auc(pred, f), "nss": nss(pred, f),
                "cc": cc(pred, d), "sim": sim(pred, d), "kld": kld(pred, d),
            },
            "empirical_baseline": {
                "ig_vs_gauss": info_gain(emp, gauss, f),
                "sauc": shuffled_auc(emp, f, other),
                "auc": auc(emp, f), "nss": nss(emp, f),
                "cc": cc(emp, d), "sim": sim(emp, d), "kld": kld(emp, d),
            },
        })
        if n % 25 == 0 or n == len(test):
            print(f"  {n}/{len(test)}")

    def avg(who: str, key: str, subset: list | None = None) -> float:
        src = rows if subset is None else subset
        vals = [r[who][key] for r in src if not np.isnan(r[who].get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n                        model     centre bias of this dataset")
    for key, label in (("ig_vs_gauss", "IG over a plain blob"),
                       ("sauc", "shuffled AUC"),
                       ("auc", "AUC-Judd"), ("nss", "NSS"),
                       ("cc", "CC"), ("sim", "SIM"), ("kld", "KLD (lower better)")):
        print(f"  {label:<22} {avg('model', key):8.4f}    {avg('empirical_baseline', key):8.4f}")
    ig_emp = avg("model", "ig_vs_empirical")
    print(f"  {'IG over that bias':<22} {ig_emp:8.4f}")

    seen = sorted({r["category"] for r in rows})
    if len(seen) > 1:
        print("\nby category, the two numbers the verdict rests on")
        print(f"  {'category':<12} {'n':>4}  {'IG over its own bias':>21}  {'sAUC':>6}"
              f"  {'sAUC of that bias':>18}")
        for c in seen:
            sub = [r for r in rows if r["category"] == c]
            print(f"  {c:<12} {len(sub):>4}  {avg('model','ig_vs_empirical',sub):>21.4f}"
                  f"  {avg('model','sauc',sub):>6.4f}"
                  f"  {avg('empirical_baseline','sauc',sub):>18.4f}")

    print("\nverdict")
    sauc_m = avg("model", "sauc")
    beats = ig_emp > 0.05 and sauc_m > 0.55
    print(f"  information gain over the dataset's own centre bias: {ig_emp:+.4f} bits/fixation")
    print(f"  shuffled AUC, where centre bias is worth 0.5:        {sauc_m:.4f}")
    print("  " + ("the map carries information about the content, beyond position"
                  if beats else
                  "NOT ACCEPTED: this map is mostly position, not content. Do not present "
                  "it as a finding."))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"n_fit": len(fit), "n_test": len(test), "rows": rows}, fh,
                      ensure_ascii=False, indent=1)
        print(f"\nwritten: {args.json}")
    return 0 if beats else 1


if __name__ == "__main__":
    sys.exit(main())
