# What may and may not be claimed

This tool produces numbers that are easy to over-sell. This file fixes the wording, so
that a report written from it survives someone checking.

## Never say

- **"N per cent accurate."** Accuracy of what, against which metric? On the public
  benchmark the same map can win on one metric and lose on another; it is a proved result
  that no single map can be best on all of them. A percentage without a named metric and a
  named baseline means nothing.
- **"Eye tracking", "where users looked", "heatmap of user attention".** Nobody looked.
  This is a model's guess about a population.
- **"Predicted CTR", "this will lift conversion by N per cent".** Nothing here predicts
  sales, and the chain from a predicted map to a purchase has three separate gaps in it.
- **"The user's gaze path", "the order in which it is read".** No trajectory is drawn, and
  the reason is in [attention.md](attention.md).
- **"Validated against MIT300"** unless the entry is actually in the public table. Several
  commercial tools claim this and are not listed.
- **An attention map of an interface, a landing page, a slide or a banner, in any form.**
  Not a hedge, a measurement: on 990 interface frames with recorded gaze this map is 0.25
  bits per fixation *worse* than one that never saw the image. The tool refuses to draw
  it. If a map of a screen reaches a report, something went wrong on purpose.

## Say instead

- "predicted attention map", "predicted density of attention"
- "share of predicted attention on this zone"
- "a model estimate, not a measurement of anyone's gaze"
- "at the width this card is shown in search results, the lettering is N px tall"
- "by colour mass this tile ranks Nth of M against its neighbours"
- "this frame is busier than the previous version of the same screen"

## The numbers behind the rules

### A blob beats most things

A blurred blob in the centre of a frame, which has not looked at the image at all, scores
**AUC 0.783** and correlation 0.446 on MIT300. The best classical model, GBVS, reaches
0.479. So any accuracy figure quoted in AUC starts from about 78, not from zero.

### What the commercial figures actually are

- The one commercial tool listed in the public MIT/Tübingen table reaches **77 per cent of
  the human ceiling by NSS**, fourteenth of forty-six models, against marketing that says
  ninety-six.
- 3M published a real validation with its method shown: their "92 per cent accurate" is a
  share of the human ceiling, and on **advertising images specifically their model scored
  worst**, 0.73 against a ceiling of 0.93.
- Several others state a benchmark they do not appear in.

This is why a percentage without a metric is refused above, and why `centre_bias_r` is
printed next to every map.

### What is actually known about attention and money

Honest measurements exist, and they are modest:

| Finding | Source |
|---|---|
| Visual features correlate with click-through at Spearman 0.10-0.21 once price and category are controlled | eBay logs, 150 000 query-item pairs, WWW 2011 |
| +3.22 per cent CTR from letting the ranking model see the image, which is not the same as changing the image | JD.com, live A/B, KDD 2020 |
| +17.5 per cent demand from professional photography, difference-in-differences on 7 711 properties | Airbnb, ICIS 2016 |

And what is not known: **there is no public measurement of what infographics on a
marketplace cover do to click-through.** Every percentage circulating on that subject comes
from agency marketing, without a sample, without control of position, price, rating or ad
spend, and usually as before-and-after rather than a split test. Both Wildberries and Ozon
shipped a built-in randomised A/B test of the main photo, which is the platforms
themselves conceding that the effect cannot be separated any other way.

So: if someone needs to know whether a card change helps, the answer is the platform's own
split test, not this tool. This tool says why a card is hard to see. That is a different
and smaller claim.

## The one-paragraph description of the tool

> It measures visual load and predicted attention on a design or a page of designs. The
> results are model estimates, not measurements of anyone's gaze. Whether a change helps
> click-through or conversion is a question for a split test.
