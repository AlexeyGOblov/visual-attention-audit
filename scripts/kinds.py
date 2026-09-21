"""What kind of frame is being measured, and what that changes.

The measures in clutter.py are the same for every image. What differs is the width
the frame is really seen at, which numbers carry meaning, which direction counts as
a defect, and what the frame should be compared against. That is what lives here.

There is no detector. The kind is declared by whoever runs the tool. A wrong guess
made silently is worse than a question asked out loud, and the published research on
cheap image-type features puts the accuracy of telling one design type from another
at 78-81 per cent, which is one frame in five read under the wrong rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Kind:
    name: str
    summary: str
    widths: tuple[tuple[int | None, str], ...]
    compare_with: str
    verdict: bool = True
    width_lock: str | None = None        # why an arbitrary --width is refused, if it is
    grid: bool = False
    whole_frame: bool = True
    colour_mass: str = "rank"            # rank | high-is-noise | not-ranked
    weak: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def display_widths(self) -> tuple[int | None, ...]:
        return tuple(w for w, _ in self.widths)


KINDS: dict[str, Kind] = {
    "marketplace_card": Kind(
        name="marketplace_card",
        summary="a product card as marketplaces show it: 3:4, artwork over a photo",
        widths=(
            (190, "grid thumbnail, phone - the size that decides whether anyone looks"),
            (240, "grid thumbnail, desktop"),
            (390, "detail view, phone"),
        ),
        compare_with="the neighbouring cards in the same search results",
        colour_mass="rank",
        notes=(
            "The 190 px run is the one that matters. Copy that carried the message at "
            "900 px often stops existing there, and no measure will rescue it.",
            "Density is not automatically a fault: a compatibility slide with twenty "
            "model numbers is dense because the job is dense.",
        ),
    ),
    "results_grid": Kind(
        name="results_grid",
        summary="a screenshot of search results, a shelf, an icon page, a competitor set",
        widths=((None, "the screenshot as captured; tiles are measured after cutting"),),
        compare_with="the tiles inside the same screenshot",
        width_lock="the screenshot is cut, not rescaled; the tiles are measured after cutting",
        grid=True,
        whole_frame=False,
        colour_mass="rank",
        weak=(
            ("feature_congestion", "measures the grid itself, not any one design"),
            ("subband_entropy", "measures the grid itself, not any one design"),
        ),
        notes=(
            "This is the only kind grid.py may cut. On anything else it finds quiet "
            "bands that mean nothing and reports a plausible ranking of nonsense.",
        ),
    ),
    "ui_screen": Kind(
        name="ui_screen",
        summary="a working screen of an application or a site",
        widths=(
            (None, "as authored - an interface is looked at in the size it was drawn"),
            (390, "the phone layout, if there is one"),
        ),
        compare_with="the previous version of the same screen",
        colour_mass="high-is-noise",
        weak=(
            ("edge_density", "tables and rules inflate it by construction, not by fault"),
        ),
        notes=(
            "Here a high score is a question, not a verdict: an operator's screen is "
            "allowed to be dense. Pair the number with what should be noticed first.",
            "Colour mass reads the other way round on an interface. A screen that wins "
            "on colour mass is a screen that shouts.",
        ),
    ),
    "landing_page": Kind(
        name="landing_page",
        summary="a full-page capture of a promo page, much taller than it is wide",
        widths=(
            (390, "phone"),
            (1280, "desktop"),
        ),
        compare_with="the first screen against the rest of the page",
        whole_frame=False,
        colour_mass="not-ranked",
        notes=(
            "One number for the whole capture averages ten screens into a figure that "
            "describes none of them. Measure the first screen on its own.",
        ),
    ),
    "banner_ad": Kind(
        name="banner_ad",
        summary="an advertisement, a poster, a creative",
        widths=(
            (300, "small placement"),
            (728, "leaderboard"),
            (1080, "social placement"),
        ),
        compare_with="the other creatives in the same batch",
        colour_mass="rank",
        notes=(
            "A banner has about a second. The blurred frame is the honest test: what "
            "survives it is the whole message the viewer receives.",
        ),
    ),
    "packaging": Kind(
        name="packaging",
        summary="the packaging of a product, shot or rendered",
        widths=(
            (None, "as authored"),
            (150, "seen down the aisle"),
        ),
        compare_with="the packs standing next to it on the shelf",
        colour_mass="rank",
        weak=(
            ("text_mass", "the small print is required by law and is not a fault here"),
        ),
    ),
    "slide": Kind(
        name="slide",
        summary="a presentation slide, usually 16:9",
        widths=(
            (1280, "the screen in the room"),
            (200, "the thumbnail in the slide list"),
        ),
        compare_with="the other slides in the same deck",
        colour_mass="not-ranked",
        notes=(
            "Subband entropy leads here: it answers how many bits one slide is asking "
            "the room to take in at once.",
        ),
    ),
    "photo": Kind(
        name="photo",
        summary="an ordinary photograph, not a designed frame",
        widths=((None, "as captured"),),
        compare_with="other photographs, and nothing else",
        verdict=False,
        width_lock="a photograph is not published at a design width; measure it as captured",
        whole_frame=True,
        colour_mass="not-ranked",
        notes=(
            "Clutter measures do not apply. A forest scores high because a forest is "
            "busy, and that is not a defect to fix. The numbers below are only good "
            "for holding one photograph against another.",
        ),
    ),
}

NAMES = tuple(KINDS)


def get(name: str) -> Kind:
    try:
        return KINDS[name]
    except KeyError:
        raise SystemExit(
            f"unknown kind {name!r}.\n" + catalogue()
        ) from None


def catalogue() -> str:
    lines = ["Declare what the frame is with --kind. There is no detector: a kind "
             "guessed wrong is read under the wrong rules, silently.", ""]
    width = max(len(n) for n in NAMES)
    for k in KINDS.values():
        lines.append(f"  {k.name:<{width}}  {k.summary}")
    return "\n".join(lines)


def describe(kind: Kind) -> list[str]:
    """What this kind changes, printed before the numbers."""
    out = [f"kind: {kind.name} - {kind.summary}"]
    sizes = ", ".join(
        f"{w} px ({why})" if w else f"native ({why})" for w, why in kind.widths
    )
    out.append(f"measured at: {sizes}")
    out.append(f"compare with: {kind.compare_with}")
    if kind.colour_mass == "high-is-noise":
        out.append("colour mass: high is the fault here, not the goal")
    elif kind.colour_mass == "not-ranked":
        out.append("colour mass: not a ranking on this kind, it competes with nothing")
    for metric, why in kind.weak:
        out.append(f"read with care: {metric} - {why}")
    if not kind.whole_frame:
        out.append("one number for the whole frame does not mean anything on this kind")
    if not kind.verdict:
        out.append("no verdict: this tool does not judge frames of this kind")
    out.extend(kind.notes)
    return out
