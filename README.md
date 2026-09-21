# visual-attention-audit

Measure how cluttered a design is, and whether it survives being seen among its
neighbours. Numbers instead of opinion.

Ships as a [Claude Code](https://claude.com/claude-code) skill, but the scripts are plain
Python and run on their own.

Three things live here:

1. **A dependency-light implementation of the Rosenholtz visual clutter measures**
   (Feature Congestion and Subband Entropy) in numpy, scipy and Pillow. The usual Python
   port needs `pyrtools`, which does not build on Windows. This one does.
2. **A grid mode** that cuts a screenshot of a grid into tiles and ranks them by what the
   eye picks up in peripheral vision, with a blurred contact sheet as evidence.
3. **A declared kind of frame.** A product card, an operator's screen and a photograph are
   not read by the same rules, and the tool will not guess which one it has. You say, and
   that answer sets the widths, the meaning of each number, and what to compare against.

## Install

```bash
git clone https://github.com/AlexeyGOblov/visual-attention-audit
cd visual-attention-audit
pip install -r requirements.txt
python scripts/clutter.py --selftest
```

As a Claude Code skill, clone it into `~/.claude/skills/` (or symlink it there) and the
skill becomes available by name.

## Use

Say what the frame is, and it is measured at the widths that kind is really seen at:

```bash
python scripts/clutter.py card.png --kind marketplace_card      # 190, 240 and 390 px
python scripts/clutter.py screen.png --kind ui_screen           # native, plus 390 px
python scripts/clutter.py slides/*.png --kind slide --json out.json --maps maps/
```

Rank a design against its neighbours:

```bash
python scripts/grid.py results.png --out result/ --kind results_grid --grid  # grid first
python scripts/grid.py results.png --out result/ --kind results_grid --mine 3
```

`--kind` is required and never guessed. It decides the widths, which numbers mean
anything, which way round a high score reads, and what the frame is compared against.
Run `python scripts/clutter.py --help` for the list of kinds.

`result/contact-sheet.png` is sharp on top, blurred below, your tile outlined. The blurred
half is what a person actually takes in before deciding whether to look closer.

Predict where the eye will go (optional, installed separately):

```bash
pip install -r requirements-attention.txt
python scripts/get_weights.py            # once: fetch MIT-licensed weights, convert
python scripts/attention.py card.png --kind marketplace_card --out result/
```

About 190 ms a frame on a CPU. No TensorFlow at run time, no GPU, no network. The model
is MSI-Net on its SALICON weights, MIT-licensed by its author; the conversion to ONNX is
checked against the original rather than assumed, and matches to 7.7e-07 per pixel.

It reports a `centre_bias_r` alongside every map, because a blurred blob in the middle of
a frame that has not looked at the image scores AUC 0.78 on the standard benchmark. If
that number is high, the map is telling you where the frame is, not what is on it.

It also knows where it works. Put in front of recorded gaze, the map gains 0.77 bits per
fixation over a centre-bias baseline on 486 e-commerce product frames, and *loses* 0.25
bits to the same baseline on 990 interface frames. So it draws a map for product imagery
and refuses for interfaces, landing pages, slides and banners, with the numbers in the
refusal. `--anyway` overrides it and stamps the output.

It draws no gaze trajectory. Numbered fixations joined by arrows are what commercial
tools show, and the evidence does not support them: on interfaces two real people agree
with each other only about 0.49 by ScanMatch, and the usual scanpath metrics rank wrong
models above the ground truth.

## What you get

| Field | Meaning |
|---|---|
| `feature_congestion` | local variability of colour, contrast and orientation |
| `subband_entropy` | bits needed to encode the frame |
| `edge_density` | crude proxy, for sanity checks |
| `colour_mass` | mean Lab chroma on a blurred frame, what survives peripheral vision |
| `text_mass` | share of the frame covered by things shaped like a line of text |
| `line_height_px` | how tall that lettering is at this width. Below 8 px nobody reads it |

## Does it work

The self-test checks the four things that can silently break:

```
1. Steerable pyramid tiles the frequency plane (squared transfers sum to 1)
   mean 0.999994, deviation from 1: mean 5.60e-06, median 6.30e-06, max 1.30e-05
2. Clutter rises with the number of items on the frame
   items   0: FC    1.248   SE  0.000
   items   8: FC    4.366   SE  1.353
   items  40: FC   10.995   SE  2.781
   items 200: FC   20.071   SE  3.648
3. Blank frame against colour noise
   FC blank 1.248 against noise 19.729
4. Text mass finds lettering, ignores a blank frame, is not fooled by noise
   lettered: mass 0.413, 7 lines of 13.0 px; blank 0.000; noise 0.000
```

The fourth check earns its place: the first version of `text_mass` returned zero on every
frame, because its threshold was the mean plus two standard deviations, which on a frame
full of edges sits above the maximum. Nothing in the output looked wrong.

The attention map has its own acceptance test against recorded gaze, `scripts/benchmark.py`.
It passes on e-commerce product frames and fails on interfaces, and the tool acts on that
rather than merely printing it.

Against an outside baseline: 23 screens of a dense internal application, each with an
independent manual count of competing visual elements made by human reviewers.

| Measure | Spearman against the manual count | p |
|---|---|---|
| Feature Congestion | 0.652 | 0.0008 |
| Subband Entropy | 0.614 | 0.0018 |
| Edge density | 0.510 | 0.013 |

Both canonical measures beat the crude proxy, as the literature predicts.

**Honest limit.** Absolute values are not verified against the reference implementation,
because it does not build on Windows and publishes no reference numbers. Use these to
compare frames with each other and to track change, not as figures that match a paper.

## What this is not

Not eye tracking. The clutter measures are analytic ones from vision science and use no
model at all; the optional attention map uses a published model trained on gaze, with its
limits written into that section. Both describe bottom-up attention in the first moments.
Neither models someone hunting for a specific part number. Pair them with a list of what
*should* be noticed first, and compare the two lists. The gap is the finding.

An earlier version used a hand-rolled saliency map built from local contrast and
saturation. It ranked a large dark heading first on every screen and failed to reproduce
findings that colour mass did reproduce. It was removed. Hand-rolled saliency is
unreliable, which is why the field uses validated models or recorded gaze - and why the
attention mode added later uses a published model with a citation rather than a fresh
guess at the same problem.

## Credit and licence

Measures published in Rosenholtz, R., Li, Y., & Nakano, L. (2007).
[Measuring visual clutter](https://doi.org/10.1167/7.2.17). *Journal of Vision*, 7(2):17.

Algorithm structure and every numeric constant follow the MIT-licensed reference port
[kargaranamir/visual-clutter](https://github.com/kargaranamir/visual-clutter) by the User
Interfaces group at Aalto University.

This project is MIT licensed ([LICENSE](LICENSE)). The upstream notice, the citation and
a list of what was changed are in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
