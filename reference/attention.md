# The predicted attention map

Mode 3. Optional, installed separately, the only part of this project that uses a neural
network.

```bash
pip install -r requirements-attention.txt
python scripts/get_weights.py          # once: fetch and convert, about 100 MB
python scripts/attention.py --selftest
python scripts/attention.py card.png --kind marketplace_card --out result/
```

About 190 ms a frame on a CPU. No TensorFlow at run time, no GPU, no network.

## Which model, and why that one

MSI-Net, SALICON weights, from
[huggingface.co/alexanderkroner/MSI-Net](https://huggingface.co/alexanderkroner/MSI-Net),
where the model card declares `license: mit`.

The choice is about permission, not accuracy. Of the models that predict attention better:

| Model | Why it is not here |
|---|---|
| DeepGaze IIE / III / MSDB | no licence at all; the `license='MIT'` line in `setup.py` is commented out, and the licence issue has sat unanswered since 2023 |
| UMSI / UMSI++ / Imp1k | "for non-commercial research purposes only", Adobe and MIT, contact required |
| SeekUI | fine-tuned from Qwen2.5-VL-3B, which is under a research licence that forbids commercial use outright |
| SUM | MIT, and the best numbers on interfaces and on e-commerce imagery, but it needs Mamba with compiled CUDA kernels; no Windows, no CPU |
| UniAR | no code published; the Google repositories in the paper return 404 |
| EyeFormer | no licence file; Python 3.6 and torch 1.7 |

So MSI-Net is not the most accurate attention model in the world. It is the most accurate
one that can be used without asking anyone's permission, and that is the constraint this
project accepted when it chose MIT.

## What the conversion did, and how that is known

The published TensorFlow 2 SavedModel is converted once to ONNX with `tf2onnx`. No
retraining, no fine-tuning, no weight touched.

The conversion is checked rather than assumed. On the author's own example image the ONNX
output matches the TensorFlow output to **7.749e-07** at the worst single pixel, with a
correlation of **1.00000000**. Reproduce it:

```bash
python scripts/get_weights.py --verify      # needs TensorFlow, see the script header
```

## Reading the output

| Field | Meaning |
|---|---|
| `peak` | where the model expects the eye to land first, as a fraction of the frame |
| `concentration` | share of predicted attention in the busiest tenth of the frame |
| `quadrants` | how attention divides between the four quarters |
| `centre_bias_r` | how much of this map is just "the middle of a frame" |

### `centre_bias_r` is the honesty check

A blurred blob in the centre of the frame that has not looked at the image at all scores
AUC 0.783 and correlation 0.446 on the standard benchmark. For scale, the best classical
non-neural model, GBVS, reaches 0.479. A map can therefore look entirely convincing while
carrying almost nothing about the design.

Above 0.8, the map is telling you where the frame is, not what is on it. Say so in the
report rather than letting a designer move elements towards the middle to satisfy an
artefact of the method.

### What the numbers looked like on real frames

Measured while building this, on three frames, same model, same settings:

| Frame | Busiest tenth holds | `centre_bias_r` |
|---|---|---|
| the author's example photograph | 80 per cent | 0.55 |
| a product card | 32 per cent | -0.01 |
| a screenshot of search results | 24 per cent | 0.53 |

The model is decisive on a photograph, which is what it was trained on, and spreads out on
designs, which it was not. Attention on the results screenshot divided 27/26/25/22 between
the quadrants, which is another way of saying it had little to tell us.

## Where it stands, by kind

`attention.py` prints this itself on every run, so that nobody has to remember which
claims were measured and which were argued.

| Kind | Standing | Map drawn? |
|---|---|---|
| `photo` | the material the model was trained on | yes |
| `marketplace_card` | **passed**: +0.77 bits per fixation over the centre bias of 486 e-commerce frames with recorded gaze, shuffled AUC 0.76 against 0.50 | yes |
| `packaging` | product imagery, the class that passed | yes |
| `ui_screen` | **failed**: 0.07 to 0.42 bits per fixation *worse* than the centre bias, across desktop, web and mobile | **refused** |
| `landing_page` | **failed**: 0.29 bits worse on web pages | **refused** |
| `slide`, `banner_ad` | **failed**: 0.23 bits worse on posters, the nearest measured thing | **refused** |
| `results_grid` | **failed**: a laid-out screen, and every laid-out screen measured lost | **refused** |

A refusal can be overridden with `--anyway`, and the output is then stamped as not for a
report. Both runs, with the per-category tables, are in [validation.md](validation.md).

The failure is not the model being weak. It is the model being *wrong in a particular
direction*: it was trained on photographs, which are looked at from the centre outwards,
and a laid-out screen is read from the top left. Its shuffled AUC on interfaces is 0.65
against the baseline's 0.53, so it does see content. The content it sees is not enough to
pay for the positional prior it brings with it.

## Two limits that are not negotiable

### On interfaces the model is outside its training

It was trained on SALICON, which its own author describes as mouse movement over natural
photographs used as a proxy for gaze. Models trained that way reproduce real gaze on
interfaces about half as well as models trained on interfaces: correlation 0.43-0.52
against 0.83 (UEyes, CHI 2023, 62 participants, 1 980 screens).

Worse than the size of the gap is its direction. Natural scenes are looked at from the
centre outwards. Interfaces are not: the same study reports a **top-left bias** and states
plainly that user interfaces are not viewed the way natural scenes are. A photographic
model therefore overstates the middle of a screen and understates its header. That is a
systematic tilt, not noise.

This applies to interfaces and not to product frames. On the e-commerce set the gaze is
centred (centre of mass 0.504 down, 0.470 across, middle third holding 45 per cent), so
the tilt argued here is not a reason to discount the map on a card. See
[validation.md](validation.md).

The transfer also fails *between* kinds of interface. A model trained on mobile screens
drops from 0.899 to 0.849 on posters, and one trained on web pages drops from 0.905 to
0.832 on mobile. This is the evidence behind `--kind` being a required question rather
than a label.

`scripts/attention.py` prints this warning itself whenever the kind is far from what the
model saw.

### There is no trajectory and there will not be one

Numbered fixations joined by arrows are the standard picture in commercial tools. The
evidence does not support drawing them.

- On interfaces, **two real people agree with each other only about 0.49 by ScanMatch**
  (ETRA 2026, leave-one-out on UEyes). There is no single order of looking to draw.
- The usual scanpath metrics reward wrong models. Kümmerer and Bethge show that incorrect
  models score *above* the ground-truth model under ScanMatch and MultiMatch, and the
  effect is largest with one predicted path against one real one, which is exactly the
  commercial demo.
- Models emit a fixed number of fixations: DeepGaze III always 11, EyeFormer always 15,
  where a person makes about 37.

What can be shown honestly is where attention is predicted to pool, and, later, whether a
goal-directed viewer is likely to reach a given element at all. Not the order of the path.

## What would close the remaining gap

For product frames the gap is measured and the map earns its place. For interfaces it is
still an argument from other people's papers.

Two pieces of work, in order:

1. **Run the same acceptance test on UEyes** (CC BY 4.0, 1 980 interface screens, 12.9 GB).
   That turns `ui_screen` and `slide` from "argued" into a number, either way.
2. **Fine-tune on UEyes and SalECI** if the answer is poor. The route is legally clear:
   UEyes CC BY 4.0, SalECI MIT, `pysaliency` MIT, MSI-Net MIT. Four clean ends, which is
   rare in this field.

Nothing ships from either without passing the test in [validation.md](validation.md).
