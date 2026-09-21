# What has been checked, and what has not

Nothing in this project is asked to be believed. This file says what was verified, how,
and where the verification stops.

## Self-tests

```bash
python scripts/clutter.py --selftest        # the measures
python scripts/attention.py --selftest      # the optional model, if installed
```

### The measures

| Check | What it proves | Expected |
|---|---|---|
| Frequency tiling | the steerable pyramid is built correctly | mean 0.999994, max deviation 1.3e-05 |
| Monotonicity | clutter rises with item count (0, 8, 40, 200 items) | FC 1.2 / 4.4 / 11.0 / 20.1 |
| Extremes | a blank frame is not like colour noise | FC 1.2 against 19.7 |
| Text mass | lettering is found, blank and noise are not | 7 lines of 13 px against 0.000 and 0.000 |

The tiling check is the one that matters: it is the risky part of the construction and it
can break silently, leaving every number plausible and wrong.

The text check earns its place. The first version of `text_mass` returned zero on every
frame, because its threshold was the mean plus two standard deviations, which on a frame
full of edges sits above the maximum. Nothing in the output looked wrong. The self-test
caught it.

### The attention model

| Check | What it proves |
|---|---|
| Loads and runs | the ONNX file is present and `onnxruntime` can execute it |
| Peak lands on the blob | the model reacts to the image, on an off-centre high-contrast patch |
| Not a centre blob | the map is not simply the middle of the frame |

### The conversion

```bash
python scripts/get_weights.py --verify
```

Runs the author's own example through both the ONNX file and the original TensorFlow model
and compares. Result: largest difference at a single pixel **7.749e-07**, correlation
**1.00000000**.

## Behaviour against an outside baseline

Across 23 screens of a dense internal application, with an independent manual count of
competing visual elements per screen made by human reviewers:

| Measure | Spearman against the manual count | p |
|---|---|---|
| Feature Congestion | 0.652 | 0.0008 |
| Subband Entropy | 0.614 | 0.0018 |
| Edge density | 0.510 | 0.013 |

The two canonical measures beat the crude proxy, which is what the literature predicts.
Feature Congestion and Subband Entropy agree with each other at Spearman 0.918 across
those screens. When they disagree, look at the frame.

## Known limits

**Absolute values are not verified against the reference implementation.** It does not
build on Windows and publishes no reference figures. The constants are unchanged, so the
measures behave the same way, but do not present a single absolute value as matching a
published figure. Compare frames with each other and track change over time.

**`text_mass` is a shape heuristic.** Not OCR, not a text detector. A row of badges can
read as a line of text. Look at the frame before quoting the number.

**The attention model is measured on product imagery, not on interfaces.** See below: it
passes on e-commerce frames and has not been put in front of interface gaze data yet.

## The acceptance test for the attention map

```bash
python scripts/benchmark.py --data data/ECdata --json result.json
```

Before the attention map may be presented as a finding it has to beat a baseline that
knows only *where people look regardless of content*. That bar is the whole point: a
blurred blob in the middle of a frame, which never saw the image, scores AUC 0.78 on the
standard benchmark, so a model can clear an impressive-looking AUC while knowing nothing.

Two baselines, and the second is the honest one: a generic centre Gaussian, and the
**empirical centre bias of the dataset itself**, built by averaging the fixations of one
half and evaluated on the other half, which it has never seen.

The bar was written into the script before any result was seen: information gain over the
empirical baseline above 0.05 bits per fixation, and shuffled AUC above 0.55.

### Result on SalECI, 21.09.2026

972 images of e-commerce product imagery with recorded gaze, MIT licensed. The empirical
baseline was fitted on 486 and the model evaluated on the other 486.

| | model | the dataset's own centre bias |
|---|---|---|
| **IG over the empirical bias, bits/fixation** | **+0.7674** | - |
| **shuffled AUC** | **0.7630** | 0.4974 |
| AUC-Judd | 0.8414 | 0.7250 |
| NSS | 1.5527 | 0.8673 |
| CC | 0.5889 | 0.3735 |
| SIM | 0.5137 | 0.3549 |
| KLD (lower is better) | 0.8636 | 1.2598 |

The baseline's shuffled AUC of 0.4974 is itself a check on the method: a map that is
nothing but the average place people look must score 0.5 there by construction, and it
does. The negatives are pooled from ten other frames rather than one, which is what pulls
that figure onto its theoretical value instead of leaving it noisy.

**Passed.** The map carries information about the content of a product frame, not only
about where the frame is.

Read the AUC-Judd row before quoting any of the others. The baseline that knows nothing
scores 0.726 there against the model's 0.841, a gap so small it would be easy to present
either number as impressive. On shuffled AUC the same baseline falls to 0.502, which is
what it deserves. This is why the verdict is not taken on AUC or on correlation.

**External cross-check.** A published table for a comparable saliency model on this same
dataset reports AUC 0.842; this pipeline gives 0.841 for MSI-Net. The split is not the
same, so this is a sanity signal rather than a replication, but it says the pipeline is
measuring the quantity it claims to measure.

### Two honest caveats on that result

- **The material is close but not identical.** SalECI is Chinese marketplace imagery,
  square, with Chinese lettering: product photo plus overlaid promotional type and price
  badges. Structurally that is our kind of card. It is not 3:4, not Cyrillic, and not
  Wildberries or Ozon.
- **On this material the centre bias really is central**: the gaze centre of mass across
  the set sits at 0.504 down and 0.470 across, and the middle third holds 45 per cent of
  it. The top-left bias reported for interfaces does not appear here. So the warning that
  a photographic model is tilted the wrong way applies to `ui_screen`, and is not
  supported for product frames.

### Result on UEyes, 21.09.2026: FAILED

1 980 interface screens, 62 participants, CC BY 4.0, four categories of 495. Same test,
same code, fixation maps at three seconds of viewing. Run twice: on the authors' own split
(1 872 fit, 108 test) and on a random half (990 fit, 990 test), because the official test
set is small and a conclusion this consequential should not rest on 27 frames per category.

| | authors' split, n=108 | random half, n=990 |
|---|---|---|
| **IG over the empirical bias** | **-0.2873** | **-0.2496** |
| shuffled AUC | 0.6500 | 0.6544 |
| shuffled AUC of the baseline | 0.5317 | 0.5311 |
| AUC-Judd, model against baseline | 0.7554 / **0.7722** | 0.7650 / **0.7745** |
| NSS, model against baseline | 1.0295 / **1.0426** | 1.0910 / 1.0736 |
| CC, model against baseline | 0.4260 / **0.4523** | 0.4499 / 0.4280 |

Per category, information gain over that category's own centre bias:

| | authors' split | random half |
|---|---|---|
| desktop | -0.0532 | -0.0672 |
| poster | -0.2751 | -0.2259 |
| web | -0.5234 | -0.2931 |
| mobile | -0.2974 | -0.4199 |

**Every category is negative in both runs.** The information gain is negative, which is
not "weak" but "worse than nothing": on a laid-out screen this map predicts where people
actually looked **less well than a map that never saw the image** and knows only where
people tend to look at that sort of screen.

The one number that still favours the model, shuffled AUC 0.65 against the baseline's
0.53, says the map is not empty: it does carry some content signal. It is just that the
signal is outweighed by the model's positional prior being wrong. It was trained on
photographs, which are looked at from the centre outwards, and interfaces are read from
the top left.

Our CC of 0.43-0.45 sits inside the 0.43-0.52 that the UEyes authors report for saliency
models not trained on interfaces. That is the second external consistency check on this
pipeline, and this time it lands on a failure rather than a pass.

### What was done about it

`scripts/attention.py` now **refuses** to draw a map for `ui_screen`, `landing_page`,
`results_grid`, `slide` and `banner_ad`, and says why. `--anyway` overrides the refusal and
stamps the output as not for a report.

`banner_ad` moved from "close to measured" to failed. The reasoning that put it near the
passing side was that a banner is product imagery with type over it, like the SalECI
frames. The nearest thing actually measured is a poster, and posters fail. An argument
from resemblance lost to a measurement, which is the correct order.

What survives: `marketplace_card`, `packaging` and `photo`. The attention map turns out to
be a tool for product frames, not for design in general.

## How to measure performance claims made by other tools

Ask which metric, then check the public table at
[saliency.tuebingen.ai](https://saliency.tuebingen.ai/results.html). Concretely: the one
commercial entry that is actually listed reaches about three quarters of the human ceiling
by NSS and sits fourteenth of forty-six, while its marketing says ninety-six per cent. The
gap is not dishonesty about the arithmetic, it is a different metric and an unstated
baseline. See [claims.md](claims.md).

## Performance

About 15 seconds for a 1440 x 900 frame with the clutter measures, a fraction of a second
at thumbnail width. The attention model adds about 190 ms a frame. For batches, measure at
display width and the whole thing becomes cheap.
