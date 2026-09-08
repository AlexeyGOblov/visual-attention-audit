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
