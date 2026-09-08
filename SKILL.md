---
name: visual-attention-audit
description: >-
  Measure a design's visual clutter and where the eye lands, with numbers instead of
  opinion. Computes the two published Rosenholtz clutter measures (Feature Congestion and
  Subband Entropy) plus colour mass on a blurred frame, and ranks a frame against its
  neighbours in a grid (search results, a shelf, an icon page, a competitor set).
  Works on any image: UI screenshot, landing page, banner, product card, packaging.
  Triggers: "is this design too busy", "measure visual clutter", "which element draws
  the eye first", "does my design stand out among competitors", "squint test",
  "visual noise score", "attention audit", "why does this screen feel cluttered",
  "compare my thumbnail against the competition". NOT an eye-tracking replacement and
  NOT a saliency model trained on gaze data; see "What this is not" below.
---

# Visual attention audit

Two questions, answered with numbers:

1. **How cluttered is this frame, and what draws the eye first?**
2. **Does it hold up among its neighbours, at the size people actually see it?**

Both run on plain images. No browser, no network, no model weights. Dependencies are
numpy, scipy and Pillow.

## Verify before you trust it

```bash
python scripts/clutter.py --selftest
```

Three checks must pass:

| Check | What it proves | Expected |
|---|---|---|
| Frequency tiling | the steerable pyramid is built correctly | mean 0.999994, max deviation 1.3e-05 |
| Monotonicity | clutter rises with item count (0, 8, 40, 200 items) | FC 1.2 / 4.4 / 11.0 / 20.1 |
| Extremes | a blank frame is not like colour noise | FC 1.2 against 19.7 |

## Mode 1: measure frames

```bash
python scripts/clutter.py frame.png [more.png ...] --width 190 --json out.json --maps maps/
```

`--width` rescales to the width the design is actually displayed at, before measuring.
This is not cosmetic. A 1200 px design shown at 190 px in a grid is a different stimulus:
text that carried the message at full size stops existing at thumbnail size. Measure at
every size the design is really seen at, not only at the size it was authored at.

Typical display widths worth measuring, in a shopping or gallery context:

| Context | Width |
|---|---|
| Grid thumbnail, phone | 170-190 px |
| Grid thumbnail, desktop | 220-260 px |
| Detail view, phone | 390 px |
| Detail view, desktop | 500-700 px |

Output per frame:

| Field | Meaning |
|---|---|
| `feature_congestion` | local variability of colour, contrast and orientation. The harder it is to add one more element that would still be noticed, the higher this is |
| `subband_entropy` | bits needed to encode the frame, that is, how much is going on |
| `edge_density` | crude proxy, useful as a sanity check |
| `colour_mass` | mean Lab chroma on a blurred frame: what survives peripheral vision |
| `components` | the colour, contrast and orientation parts of Feature Congestion |

`--maps` writes a local clutter map per frame, so you can see *where* the clutter is.

## Mode 2: rank against neighbours

A design is rarely seen alone. Give the tool one screenshot of the grid it lives in.

```bash
python scripts/grid.py screenshot.png --out result/ --grid          # check detection first
python scripts/grid.py screenshot.png --out result/ --mine 3
python scripts/grid.py screenshot.png --out result/ --mine 3 --cols 5 --rows 3
python scripts/grid.py screenshot.png --out result/ --mine 3 --full
```

Always run `--grid` first: it prints the detected grid without cutting anything. If the
column or row count is wrong, force it with `--cols` and `--rows`.

You get the cut tiles, a `grid.json` with each tile's rank, and `contact-sheet.png`:
sharp on top, blurred below, your tile outlined. The blurred half is the point. It shows
what a person takes in during the fraction of a second before they decide to look closer.

The headline number is your tile's rank by colour mass. If it sits in the bottom half,
the design loses the grid no matter how good it looks on its own.

## How to read the numbers

- **A high score is not automatically a defect.** A page of twenty compatibility badges is
  dense because the job is dense. Clutter becomes a finding only together with an answer
  to "what does this element tell the reader", and the price of removing it.
- **Compare, do not grade.** There is no universal threshold. Compare variants of the same
  design, the same design at different sizes, or your design against its neighbours.
- **Two measures, one story.** Feature Congestion and Subband Entropy agree closely
  (Spearman 0.918 across 23 test screens). When they disagree, look at the frame.

## What this is not

- **Not eye tracking.** Eye tracking is ground truth; this predicts one part of it.
- **Not a gaze-trained saliency model.** Deep saliency predictors and commercial tools are
  trained on recorded gaze. These are analytic measures from the vision-science literature.
- **Not a model of goal-driven search.** The measures describe bottom-up attention in the
  first moments. Someone hunting for a specific part number behaves differently. Pair the
  numbers with a list of what *should* be noticed first, and compare the two.
- **Deliberately not a hand-rolled saliency map.** An earlier version of this work used a
  local-contrast-and-saturation saliency map. It ranked a large dark heading first on
  every screen and failed to reproduce findings that colour mass reproduced. Hand-rolled
  saliency is unreliable; that is exactly why the field uses validated models or gaze data.

## Validation

**Implementation.** The self-test above verifies the steerable pyramid tiles the frequency
plane to within 1.3e-05, which is the risky part of the construction.

**Behaviour.** Across 23 screens of a dense internal application, with an independent
manual count of competing visual elements per screen made by human reviewers:

| Measure | Spearman against the manual count | p |
|---|---|---|
| Feature Congestion | 0.652 | 0.0008 |
| Subband Entropy | 0.614 | 0.0018 |
| Edge density | 0.510 | 0.013 |

The two canonical measures beat the crude proxy, which is what the literature predicts.

**Known limit.** Absolute values are not verified against the reference implementation:
it does not build on Windows and publishes no reference numbers. The measures are sound
for comparing frames with each other and for tracking change over time. Do not present a
single absolute value as matching a published figure.

## Performance

About 15 seconds for a 1440 x 900 frame; a fraction of a second at thumbnail width. For
batches, measure at display width and the whole thing becomes cheap.

## Files

| File | Role |
|---|---|
| `scripts/clutter.py` | the measures, colour space, steerable pyramid, self-test |
| `scripts/grid.py` | grid detection, tile cutting, ranking, contact sheet |

## Credit

Measures published by Rosenholtz, Li and Nakano (Journal of Vision, 2007). Constants and
algorithm structure follow the MIT-licensed reference port by the User Interfaces group at
Aalto University. See THIRD-PARTY-NOTICES.md for the upstream notice and the citation.
