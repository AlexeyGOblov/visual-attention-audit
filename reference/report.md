# The report

Every run ends with one report in this shape, so that whoever reads it, a person or another
model, can tell what was measured, how far to trust it and what to do next, without having
been in the session. Write it in the language of the person who asked.

The report is a folder: `report.md` plus the frames it shows, as JPEG, next to it. Markdown
because any model reads it as plain text and any repository host renders it as a page.

## Layout

```markdown
# <What was measured>, <date>

<One sentence: which frames, which kind, what question.>

## First

<Three lines at most: the findings to act on, most severe first, each with its fix.>

## <Section: one per mode that ran>

### Done
<The exact command, the input frames, the --kind, the widths it measured at.>

### Checked
<Self-test result. Whether the attention model passed acceptance on this kind.
What the measure cannot see here.>

### Result
<The numbers, as a table, and the frame that shows them.>

### Recommendations
| Finding | Severity | Fix | What the fix costs |
|---|---|---|---|

## Not checked

<What was not run or not measured, and why.>

## Decisions for the owner

<Each open choice with a recommendation and a two-word answer.>
```

Sections, in this order, only those that ran: **Neighbours** (`grid.py`), **Clutter and
lettering** (`clutter.py`), **Predicted attention** (`attention.py`), **Against the brief**
(the list of what should be noticed first, compared with what the numbers say is noticed).

## Rules

- **Every number names its frame and its width.** "Rank 8 of 8 by ink" without saying
  which tile and which screenshot is not a result.
- **Say which lines were judged by eye.** Where a finding rests on looking at an overlay
  rather than on a number, mark it: *(by eye)*. The reader has to be able to tell measure
  from opinion.
- **A finding carries its price.** What the reader of the design does not see, and what
  is lost when the element is removed. "Cluttered" alone is rejected.
- **Severity has four steps:** blocking (the buyer leaves or picks the wrong thing), high
  (the main point is missed in a second), medium (slows reading), low (roughness).
- **Frames in the report are the ones measured,** not re-rendered for looks: the contact
  sheet, the overlay, the frame at the width that mattered.
- **No figure without its caveat when it has one.** Attention maps are predictions, not
  gaze. `line_height_px` is a shape heuristic. Say so where the number is quoted.
- **Keep it short.** A section that found nothing says so in one line.
