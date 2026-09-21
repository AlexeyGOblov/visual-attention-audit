# -*- coding: utf-8 -*-
"""Grid mode: cut a screenshot of a grid into tiles and rank them by what draws the eye.

A design is rarely seen alone. A product card sits in a row of competitors, an app icon
sits on a page of icons, a pack sits on a shelf. This measures a frame where it actually
lives: among its neighbours, at the size they are shown, through peripheral vision.

What it does:
  1. Finds the grid by projecting variance (quiet strips are the gutters).
  2. Cuts the tiles and writes them out.
  3. Scores each tile by colour mass and ink mass on a blurred frame.
  4. Builds a contact sheet: sharp on top, blurred below, yours outlined.
  5. Prints where your tile ranks among its neighbours.

Usage:
    python grid.py screenshot.png --out dir [--mine 3] [--blur 2.5]
    python grid.py screenshot.png --out dir --grid              # show detected grid only
    python grid.py screenshot.png --out dir --cols 5 --rows 3   # force the grid
    python grid.py screenshot.png --out dir --mine 3 --full     # add clutter measures
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clutter import edge_density, feature_congestion, rgb_to_lab, subband_entropy  # noqa: E402
import kinds  # noqa: E402


# ------------------------------------------------------------------ grid detection

def _quiet_runs(profile: np.ndarray, min_run: int, quiet: float) -> list:
    """Long stretches where the image barely changes: candidate gutters."""
    mask = profile <= quiet
    runs, start = [], None
    for i, q in enumerate(mask):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= min_run:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_run:
        runs.append((start, len(mask)))
    return runs


def find_grid(rgb: np.ndarray, min_tile: int = 60, quiet_quantile: float = 0.12):
    g = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float64)
    col_gaps = _quiet_runs(g.std(axis=0), max(4, min_tile // 12),
                           np.quantile(g.std(axis=0), quiet_quantile))
    row_gaps = _quiet_runs(g.std(axis=1), max(4, min_tile // 12),
                           np.quantile(g.std(axis=1), quiet_quantile))

    def cuts(gaps, size):
        c = [0]
        for a, b in gaps:
            mid = (a + b) // 2
            if mid - c[-1] >= min_tile:
                c.append(mid)
        if size - c[-1] >= min_tile:
            c.append(size)
        else:
            c[-1] = size
        return c

    return cuts(col_gaps, rgb.shape[1]), cuts(row_gaps, rgb.shape[0])


def even_grid(rgb: np.ndarray, cols: int, rows: int):
    return ([round(i * rgb.shape[1] / cols) for i in range(cols + 1)],
            [round(i * rgb.shape[0] / rows) for i in range(rows + 1)])


# ------------------------------------------------------------------ tile scores

def colour_mass(rgb: np.ndarray, blur_px: float) -> float:
    """Mean Lab chroma on a blurred tile: colour that survives peripheral vision."""
    im = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_px))
    lab = rgb_to_lab(np.asarray(im))
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def ink_mass(rgb: np.ndarray, blur_px: float) -> float:
    """How much darker than white the blurred tile is, on average."""
    im = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_px))
    return float((255.0 - np.asarray(im.convert("L"), dtype=np.float64)).mean())


# ------------------------------------------------------------------ contact sheet

def contact_sheet(tiles: list, path: str, blur_px: float, mine: int | None, per_row: int):
    if not tiles:
        return
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + per_row - 1) // per_row
    pad = 6
    w = per_row * (tw + pad) + pad
    h = rows * (th + pad) + pad
    sheet = Image.new("RGB", (w, h), (232, 236, 240))
    for i, t in enumerate(tiles):
        r, c = divmod(i, per_row)
        sheet.paste(Image.fromarray(t.astype(np.uint8)), (pad + c * (tw + pad), pad + r * (th + pad)))
    both = Image.new("RGB", (w, h * 2 + 12), (232, 236, 240))
    both.paste(sheet, (0, 0))
    both.paste(sheet.filter(ImageFilter.GaussianBlur(blur_px)), (0, h + 12))
    if mine is not None and 0 <= mine < len(tiles):
        r, c = divmod(mine, per_row)
        box = (pad + c * (tw + pad) - 3, pad + r * (th + pad) - 3,
               pad + c * (tw + pad) + tw + 2, pad + r * (th + pad) + th + 2)
        d = ImageDraw.Draw(both)
        d.rectangle(box, outline=(200, 30, 60), width=3)
        d.rectangle((box[0], box[1] + h + 12, box[2], box[3] + h + 12), outline=(200, 30, 60), width=3)
    both.save(path)


# ------------------------------------------------------------------ driver

def run(args) -> int:
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    if args.cols and args.rows:
        xs, ys = even_grid(rgb, args.cols, args.rows)
    else:
        xs, ys = find_grid(rgb, args.min_tile)
    print(f"grid: {len(xs) - 1} columns x {len(ys) - 1} rows "
          f"(column cuts {xs}, row cuts {ys})")
    if args.grid:
        return 0

    os.makedirs(args.out, exist_ok=True)
    tiles, meta = [], []
    for ri in range(len(ys) - 1):
        for ci in range(len(xs) - 1):
            tile = rgb[ys[ri]:ys[ri + 1], xs[ci]:xs[ci + 1]]
            if tile.shape[0] < args.min_tile or tile.shape[1] < args.min_tile:
                continue
            idx = len(tiles)
            tiles.append(tile)
            name = f"tile-{idx:02d}-r{ri}c{ci}.png"
            Image.fromarray(tile.astype(np.uint8)).save(os.path.join(args.out, name))
            meta.append({"index": idx, "row": ri, "column": ci, "file": name,
                         "size": [int(tile.shape[1]), int(tile.shape[0])]})

    print(f"tiles found: {len(tiles)}")
    for m, tile in zip(meta, tiles):
        m["colour_mass"] = round(colour_mass(tile, args.blur), 3)
        m["ink_mass"] = round(ink_mass(tile, args.blur), 2)
        m["edge_density"] = round(edge_density(tile), 4)
        if args.full:
            fc, _, _ = feature_congestion(tile)
            m["feature_congestion"] = round(fc, 3)
            m["subband_entropy"] = round(subband_entropy(tile), 3)

    for place, m in enumerate(sorted(meta, key=lambda x: -x["colour_mass"]), 1):
        m["rank_colour"] = place
    for place, m in enumerate(sorted(meta, key=lambda x: -x["ink_mass"]), 1):
        m["rank_ink"] = place

    sheet = os.path.join(args.out, "contact-sheet.png")
    contact_sheet(tiles, sheet, args.blur, args.mine, args.per_row or max(1, len(xs) - 1))
    print(f"contact sheet: {sheet}")

    print("\nrank by colour mass on the blurred sheet:")
    for m in sorted(meta, key=lambda x: x["rank_colour"])[:args.top]:
        mark = "  <- yours" if args.mine is not None and m["index"] == args.mine else ""
        print(f"  {m['rank_colour']:2d}. tile {m['index']:2d}  colour {m['colour_mass']:6.2f}  "
              f"ink {m['ink_mass']:6.2f}{mark}")
    if args.mine is not None:
        mine = next((m for m in meta if m["index"] == args.mine), None)
        if mine:
            print(f"\nyours: rank {mine['rank_colour']} of {len(meta)} by colour, "
                  f"{mine['rank_ink']} by ink")

    out_json = os.path.join(args.out, "grid.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"source": os.path.basename(args.image),
                   "grid": {"columns": len(xs) - 1, "rows": len(ys) - 1},
                   "blur": args.blur, "mine": args.mine, "tiles": meta}, fh,
                  ensure_ascii=False, indent=1)
    print(f"written: {out_json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rank a frame against its neighbours in a grid",
        epilog=kinds.catalogue(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("image", help="screenshot of the grid")
    ap.add_argument("--kind", choices=kinds.NAMES, metavar="KIND",
                    help="what the image is; only results_grid can be cut")
    ap.add_argument("--out", required=True, help="directory for tiles and results")
    ap.add_argument("--mine", type=int, default=None, help="index of your tile after cutting")
    ap.add_argument("--cols", type=int, default=None, help="force the number of columns")
    ap.add_argument("--rows", type=int, default=None, help="force the number of rows")
    ap.add_argument("--min-tile", type=int, default=60, help="smallest tile side, px")
    ap.add_argument("--blur", type=float, default=2.5, help="blur radius at display size, px")
    ap.add_argument("--per-row", type=int, default=None, help="tiles per contact sheet row")
    ap.add_argument("--top", type=int, default=12, help="how many ranks to print")
    ap.add_argument("--full", action="store_true", help="also compute clutter measures (slow)")
    ap.add_argument("--grid", action="store_true", help="only report the detected grid")
    args = ap.parse_args()

    if not args.kind:
        ap.error("--kind is required. Grid mode cuts on quiet strips, and on anything "
                 "that is not a grid it finds strips anyway and ranks nonsense "
                 "convincingly.\n\n" + kinds.catalogue())
    kind = kinds.get(args.kind)
    if not kind.grid:
        ap.error(f"grid mode does not apply to {kind.name} ({kind.summary}). "
                 f"It is for results_grid: one screenshot holding a frame and its "
                 f"neighbours. To place a single card among competitors, capture the "
                 f"search results and pass that.")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
