# What the kind of frame changes

`--kind` is required on every command and is never guessed. This file is why.

## Why there is no detector

A detector was considered and rejected. Telling a photograph from something drawn on a
screen is reliable on cheap features, around 92-95 per cent in the published work on
image-type classification. Telling one *design* type from another on the same features is
78-81 per cent. That is one frame in five measured at the wrong widths, compared against
the wrong neighbours, and reported in the wrong words, with nothing on the screen saying
so. A question asked out loud costs a second. A wrong guess made quietly costs the whole
report, and nobody notices.

So the tool asks. If the person who brought the image is not sure, look at the image and
tell them what you think it is, then use what they confirm.

## The kinds

### `marketplace_card`

A product card as marketplaces show it: 3:4, artwork over a photo.

- **Measured at 190, 240 and 390 px.** 190 is the one that decides everything: that is the
  size in a phone's search results, where the shopper chooses whether to look at all.
- **Compared with** the neighbouring cards in the same search results, which means a
  screenshot of the results and mode 2, not the card on its own.
- **Colour mass is a rank.** Low means losing the row.
- Watch `line_height_px` at 190. Copy that carried the message at 900 px routinely does
  not exist there, and no other number will rescue it.
- Density is not automatically a fault. A compatibility slide with twenty model numbers is
  dense because the job is dense.

### `results_grid`

A screenshot of search results, a shelf, an icon page, a competitor set.

- **Not rescaled.** The screenshot is cut, and the tiles are measured after cutting.
- **The only kind mode 2 may cut.** On anything else the grid detector still finds quiet
  strips, still cuts, and still produces a convincing ranking of nonsense.
- Feature Congestion and Subband Entropy over the whole frame measure the grid, not any
  one design in it. Read the per-tile numbers instead.

### `ui_screen`

A working screen of an application or a site.

- **Measured at its native width**, plus 390 px if there is a phone layout. An interface is
  used at the size it was drawn; shrinking it invents a stimulus nobody sees.
- **Compared with** the previous version of the same screen.
- **Colour mass reads the other way round here.** On a card, losing on colour mass means
  losing the shelf. On an interface, winning on colour mass means the screen shouts.
- `edge_density` is inflated by construction: tables and rules make edges, and that is the
  work, not a defect.
- A high score is a question, not a verdict. Pair it with a list of what *should* be
  noticed first. The gap between that list and the numbers is the finding.

### `landing_page`

A full-page capture of a promo page, much taller than it is wide.

- Measured at 390 and 1280 px.
- One number for the whole capture averages ten screens into a figure that describes none
  of them. Measure the first screen on its own.

### `banner_ad`

An advertisement, a poster, a creative.

- Measured at 300, 728 and 1080 px, or at whatever the real placement is.
- **Compared with** the other creatives in the same batch, on one fixed scale. Normalising
  each creative separately makes a weak one look as strong as a good one.
- A banner has about a second. The blurred frame is the honest test: what survives it is
  the whole message the viewer receives.

### `packaging`

The packaging of a product, shot or rendered.

- Measured at its native size and at 150 px, which is the pack seen down the aisle.
- **Compared with** the packs standing next to it on the shelf.
- `text_mass` is not a fault here. The small print is required by law.

### `slide`

A presentation slide, usually 16:9.

- Measured at 1280 px, the screen in the room, and 200 px, the thumbnail in the slide list.
- **Compared with** the other slides in the same deck.
- Subband entropy leads: it answers how many bits one slide asks the room to take in at
  once.

### `photo`

An ordinary photograph, not a designed frame.

- Measured as captured. `--width` is refused: a photograph is not published at a design
  width.
- **No verdict.** A forest scores high because a forest is busy, and that is not a defect
  to fix. The numbers are good only for holding one photograph against another.
- Note the inversion: this is the kind where the clutter measures refuse to judge and the
  attention model of mode 3 is most at home, because that is what it was trained on.

## The three switches, in one place

1. **Mode 2 runs on `results_grid` and nothing else.**
2. **Colour mass changes sign** between a frame that competes for a glance (card, banner,
   pack) and a frame that should be quiet (interface, slide).
3. **`photo` gets no verdict at all.** This is the single thing that cannot be had without
   knowing the kind, and it is most of the reason the kind is asked for.

## Widths, and when `--width` is refused

Each kind carries the widths it is really seen at, and the frame is measured at all of
them. This is the heaviest correction the kind makes: before it existed the width was
typed by hand and usually forgotten, so cards were measured at the size they were drawn,
which is the most flattering size and the one no shopper ever sees.

`--width` overrides the list where that means something. It is refused on `photo` and on
`results_grid`, and the refusal says which of the two reasons applies.

## Adding a kind

Add it to `scripts/kinds.py` and nowhere else. Every command reads that table, so a new
kind arrives in all of them at once, with its widths, its sign, its weak metrics and its
basis for comparison. If a kind needs code outside that file, the design has slipped.
