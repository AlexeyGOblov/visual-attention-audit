# Third-party notices

`scripts/clutter.py` is a derivative work. Its algorithm structure and every numeric
constant follow the reference Python port of the Rosenholtz clutter measures:

**visual-clutter** — https://github.com/kargaranamir/visual-clutter

```
MIT License

Copyright (c) 2021-present, User Interfaces group, Aalto University, Finland

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

That port in turn implements the measures published in:

> Rosenholtz, R., Li, Y., & Nakano, L. (2007). Measuring visual clutter.
> *Journal of Vision*, 7(2):17, 1-22. https://doi.org/10.1167/7.2.17

## What changed here

The reimplementation replaces the `pyrtools` and `opencv-python` dependencies with
numpy, scipy and Pillow, so that the measures can be computed on Windows, where the
reference port does not build. This required writing the Gaussian pyramid, the
frequency-domain steerable pyramid, the separable convolutions and the CIELab
conversion directly.

Constants are unchanged: three pyramid levels, colour and contrast pooling sigma 3,
`deltaL2` / `deltaa2` / `deltab2` of 0.0007² / 0.1² / 0.05², combination weights
0.2088 / 0.0660 / 0.0269, four orientations, chrominance weight 0.0625, and the
square-root-of-N binning of the entropy estimator.

Because the constants are unchanged, the measures behave the same way. Absolute values
have not been compared against the reference implementation, since it does not run on
Windows and publishes no reference figures; see the README for what that means in
practice.

---

# MSI-Net, used by the optional attention mode

`scripts/attention.py` runs pre-trained weights that are not part of this project and are
not distributed with it. `scripts/get_weights.py` fetches them at the user's request from:

**MSI-Net** — https://huggingface.co/alexanderkroner/MSI-Net
(model card declares `license: mit`; the source repository
https://github.com/alexanderkroner/saliency carries an MIT `LICENSE.md`)

```
MIT License

Copyright (c) 2019 Alexander Kroner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The author asks that the following be cited when the model is used:

> Kroner, A., Senden, M., Driessens, K., & Goebel, R. (2020). Contextual encoder-decoder
> network for visual saliency prediction. *Neural Networks*, 129, 261-270.
> https://doi.org/10.1016/j.neunet.2020.05.004

## What was done to it

Nothing but a format change. The published TensorFlow 2 SavedModel is converted once to
ONNX with `tf2onnx`, so that prediction needs only `onnxruntime`. No retraining, no
fine-tuning, no change to any weight.

The conversion is verified rather than assumed: on the author's own example image the
ONNX output matches the TensorFlow output to 7.749e-07 at the worst single pixel, with a
correlation of 1.00000000. Reproduce it with `python scripts/get_weights.py --verify`.

The published weights are the SALICON base model. As the author states, SALICON records
mouse movement over natural photographs as a proxy for gaze. The variants fine-tuned on
eye-tracking data, including the `fiwi` web-page variant, are not published on that model
card and are not used here.
