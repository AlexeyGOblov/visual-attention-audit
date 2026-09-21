---
name: visual-attention-audit
description: >-
  Measure a design with numbers instead of opinion: how cluttered a frame is, whether it
  survives among its neighbours at the size people really see it, and, optionally, where a
  gaze-trained model predicts the eye will go. Works on a product card, a search-results
  screenshot, an app screen, a landing page, a banner, packaging, a slide or a photograph,
  and asks which of those it is before measuring anything.
  Triggers: "is this design too busy", "measure visual clutter", "what draws the eye
  first", "does my card stand out among competitors", "squint test", "why does this screen
  feel cluttered", "where will the eye go on this", "compare these design variants".
  Not eye tracking, and it draws no gaze trajectory.
---

# Visual attention audit

Three questions, answered with numbers:

1. **How cluttered is this frame, and does its lettering still exist at the size it is shown?**
2. **Does it hold up among its neighbours?**
3. **Where does a model trained on gaze expect the eye to go?** (optional)

The first two run on plain images: no browser, no network, no model weights, just numpy,
scipy and Pillow. The third is installed separately.

## First: verify, then say what the frame is

```bash
python scripts/clutter.py --selftest
```

Four checks must pass. If they do not, nothing below means anything. Details and the
expected figures: [reference/validation.md](reference/validation.md).

Then every command needs `--kind`. **It is never guessed.** Ask whoever brought the image;
if they are unsure, look at it, say what you think it is, and use what they confirm.

| `--kind` | What it is | Measured at | Colour mass |
|---|---|---|---|
| `marketplace_card` | a product card, 3:4, artwork over a photo | 190, 240, 390 px | rank against neighbours |
| `results_grid` | a screenshot of results, a shelf, a competitor set | as captured, then cut | rank of the tiles |
| `ui_screen` | a working screen of an app or a site | native, plus 390 px | **high is the fault** |
| `landing_page` | a full-page capture, much taller than wide | 390, 1280 px | not a ranking |
| `banner_ad` | an advertisement, a poster, a creative | 300, 728, 1080 px | rank in the batch |
| `packaging` | the packaging of a product | native, plus 150 px | rank on the shelf |
| `slide` | a presentation slide | 1280, 200 px | not a ranking |
| `photo` | an ordinary photograph | as captured | **no verdict at all** |

The kind decides the widths, which numbers mean anything, which way round a high score
reads, and what the frame is compared against. Why it is asked rather than detected, and
what each kind changes in full: [reference/kinds.md](reference/kinds.md).

## Mode 1: measure frames

```bash
python scripts/clutter.py card.png --kind marketplace_card --json out.json
python scripts/clutter.py screen.png --kind ui_screen --maps maps/
python scripts/clutter.py slides/*.png --kind slide
```

Each frame is measured at every width its kind calls for.

| Field | Meaning |
|---|---|
| `feature_congestion` | local variability of colour, contrast and orientation: how hard it is to add one more element that would still be noticed |
| `subband_entropy` | bits needed to encode the frame, that is, how much is going on |
| `edge_density` | crude proxy, useful as a sanity check |
| `colour_mass` | mean Lab chroma on a blurred frame: what survives peripheral vision |
| `text_mass`, `lines` | share of the frame covered by things shaped like lines of text |
| `line_height_px` | how tall that lettering is **at this width**. Below 8 px nobody reads it |
| `components` | the colour, contrast and orientation parts of Feature Congestion |

`line_height_px` answers the question no attention model can, because every one of them
shrinks the frame to about 300 px before it looks at anything: at the width this design is
really shown, does its lettering still exist. It is a shape heuristic, so a row of badges
can read as a line of text. Look at the frame before quoting it.

`--maps` writes a local clutter map per frame, so you can see *where* the clutter is.

## Mode 2: rank against neighbours

A design is rarely seen alone. Give the tool one screenshot of the grid it lives in.

```bash
python scripts/grid.py shot.png --out result/ --kind results_grid --grid   # check first
python scripts/grid.py shot.png --out result/ --kind results_grid --mine 3
python scripts/grid.py shot.png --out result/ --kind results_grid --mine 3 --cols 5 --rows 3
```

Always run `--grid` first: it prints the detected grid without cutting anything. If the
column or row count is wrong, force it with `--cols` and `--rows`.

You get the cut tiles, a `grid.json` with each tile's rank, and `contact-sheet.png`: sharp
on top, blurred below, your tile outlined. The blurred half is the point. It is what a
person takes in in the fraction of a second before deciding whether to look closer.

The headline is your tile's rank by colour mass. In the bottom half, the design loses the
row no matter how good it looks on its own.

## Mode 3: predicted attention map (optional)

```bash
pip install -r requirements-attention.txt
python scripts/get_weights.py                    # once: fetch and convert, ~100 MB
python scripts/attention.py --selftest
python scripts/attention.py card.png --kind marketplace_card --out result/
```

MSI-Net on its SALICON weights, MIT-licensed by its author, converted once to ONNX so that
prediction needs only `onnxruntime`. About 190 ms a frame on a CPU.

Read `centre_bias_r` before anything else: a blurred blob in the middle of the frame that
never looked at the image scores AUC 0.78 on the standard benchmark, so a map can look
convincing and carry nothing. Above 0.8, it is telling you where the frame is.

**It works on product frames and is refused on laid-out screens, and both were measured.**
Against 486 e-commerce frames with recorded gaze the map gains 0.77 bits per fixation over
knowing only where people tend to look. Against 990 interface frames it *loses* 0.25 bits
to the same kind of baseline: worse than a map that never saw the image. So it draws for
`marketplace_card`, `packaging` and `photo`, and refuses for `ui_screen`, `landing_page`,
`results_grid`, `slide` and `banner_ad`, saying why. `--anyway` overrides and stamps the
output as not for a report.

**No trajectory is drawn**: on interfaces two real people agree with each other only about
0.49 by ScanMatch, so there is no single order of looking to draw. The full case, the
provenance of the weights and why better models were rejected:
[reference/attention.md](reference/attention.md). The acceptance run:
[reference/validation.md](reference/validation.md).

## How to read the numbers

- **A high score is not automatically a defect.** Twenty compatibility badges are dense
  because the job is dense. Clutter becomes a finding only together with an answer to
  "what does this element tell the reader", and the price of removing it.
- **Compare, do not grade.** There is no universal threshold. Compare variants of one
  design, one design at different sizes, or a design against its neighbours.
- **Bring the list of what *should* be noticed first.** The gap between that list and what
  the numbers say is noticed is the finding. Without the list there is no finding, only
  arithmetic.

## Last: the report

Every run ends with a report folder, `report.md` and the frames it shows: what was done,
how it was checked, the result and the recommendations, for each mode that ran. The
template and its rules: [reference/report.md](reference/report.md). Where the report is
published is the caller's business, not this skill's.

## What this is not

- **Not eye tracking.** Where a tool claims a percentage of accuracy, ask which metric and
  against which baseline. The wording this project will and will not use, with the figures
  behind it: [reference/claims.md](reference/claims.md).
- **Not a predictor of clicks or sales.** There is no public measurement of what a cover
  design does to click-through on any marketplace; the platforms' own split tests are the
  only honest answer to that question.
- **Not a model of goal-driven search.** These measures describe bottom-up attention in
  the first moments. Someone hunting for a part number behaves differently.
- **Deliberately not a hand-rolled saliency map.** An earlier version used one built from
  local contrast and saturation. It ranked a large dark heading first on every screen and
  failed to reproduce findings that colour mass reproduced. It was removed. That is also
  why mode 3 uses a published model with a citation rather than a fresh guess.

## Files

| File | Role |
|---|---|
| `scripts/clutter.py` | the measures, colour space, steerable pyramid, self-test |
| `scripts/grid.py` | grid detection, tile cutting, ranking, contact sheet |
| `scripts/kinds.py` | the one table of what each kind changes |
| `scripts/attention.py` | the optional predicted attention map |
| `scripts/get_weights.py` | one-time fetch and conversion of the weights, with a check |
| `scripts/benchmark.py` | the acceptance test: does the map beat knowing only where the frame is |
| `reference/kinds.md` | why the kind is asked, and what each one changes in full |
| `reference/attention.md` | the model, its provenance, its limits, why no trajectory |
| `reference/validation.md` | every check that was run, and where checking stops |
| `reference/claims.md` | allowed and forbidden wording, with the evidence |
| `reference/report.md` | the one shape every report takes |

## Credit

Measures published by Rosenholtz, Li and Nakano (Journal of Vision, 2007). Constants and
algorithm structure follow the MIT-licensed reference port by the User Interfaces group at
Aalto University. The attention model is MSI-Net (Kroner and others, Neural Networks,
2020). See THIRD-PARTY-NOTICES.md for both notices and the citations.
