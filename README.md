# visual-attention-audit

Measure how cluttered a design is, and whether it survives being seen among its
neighbours. Numbers instead of opinion.

Ships as a [Claude Code](https://claude.com/claude-code) skill, but the scripts are plain
Python and run on their own.

Two things live here:

1. **A dependency-light implementation of the Rosenholtz visual clutter measures**
   (Feature Congestion and Subband Entropy) in numpy, scipy and Pillow. The usual Python
   port needs `pyrtools`, which does not build on Windows. This one does.
2. **A grid mode** that cuts a screenshot of a grid into tiles and ranks them by what the
   eye picks up in peripheral vision, with a blurred contact sheet as evidence.

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

Measure frames, at the width they are actually displayed at:

```bash
python scripts/clutter.py design.png --width 390
python scripts/clutter.py slides/*.png --width 190 --json out.json --maps maps/
```

Rank a design against its neighbours:

```bash
python scripts/grid.py search-results.png --out result/ --grid   # check the grid first
python scripts/grid.py search-results.png --out result/ --mine 3
```

`result/contact-sheet.png` is sharp on top, blurred below, your tile outlined. The blurred
half is what a person actually takes in before deciding whether to look closer.

## What you get

| Field | Meaning |
|---|---|
| `feature_congestion` | local variability of colour, contrast and orientation |
| `subband_entropy` | bits needed to encode the frame |
| `edge_density` | crude proxy, for sanity checks |
| `colour_mass` | mean Lab chroma on a blurred frame, what survives peripheral vision |

## Does it work

The self-test checks the three things that can silently break:

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
```

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

Not eye tracking, and not a saliency model trained on gaze. These are analytic measures
from vision science. They describe bottom-up attention in the first moments; they do not
model someone hunting for a specific thing. Pair them with a list of what *should* be
noticed first, and compare the two lists. The gap is the finding.

An earlier version used a hand-rolled saliency map built from local contrast and
saturation. It ranked a large dark heading first on every screen and failed to reproduce
findings that colour mass did reproduce. It was removed. Hand-rolled saliency is
unreliable, which is why the field uses validated models or recorded gaze.

## Credit and licence

Measures published in Rosenholtz, R., Li, Y., & Nakano, L. (2007).
[Measuring visual clutter](https://doi.org/10.1167/7.2.17). *Journal of Vision*, 7(2):17.

Algorithm structure and every numeric constant follow the MIT-licensed reference port
[kargaranamir/visual-clutter](https://github.com/kargaranamir/visual-clutter) by the User
Interfaces group at Aalto University.

This project is MIT licensed ([LICENSE](LICENSE)). The upstream notice, the citation and
a list of what was changed are in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
