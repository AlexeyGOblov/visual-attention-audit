# -*- coding: utf-8 -*-
"""Visual clutter measures: Feature Congestion and Subband Entropy.

Rosenholtz, R., Li, Y., & Nakano, L. (2007). Measuring visual clutter.
Journal of Vision, 7(2):17, 1-22.

This is a dependency-light reimplementation. It follows the reference Python port
`kargaranamir/visual-clutter` (MIT, Aalto University User Interfaces group) constant for
constant, but replaces its `pyrtools` and `opencv` dependencies with plain numpy, scipy
and Pillow, so it runs on Windows where the reference port cannot be built.

Verify the implementation before trusting a number:

    python clutter.py --selftest

Usage:

    python clutter.py frame.png [more.png ...] [--width 190] [--json out.json] [--maps dir]

`--width` rescales each frame to the width it is actually displayed at before measuring.
This matters: a 1200 px design shown at 190 px in a grid is a different stimulus, and
clutter has to be measured at the size the viewer actually sees.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, signal

# ------------------------------------------------------------------ colour space

def rgb_to_lab(im: np.ndarray) -> np.ndarray:
    """sRGB (0-255) -> CIELab, following the reference port's conversion.

    Note: the reference divides XYZ by 95.047/100/108.833 while XYZ from linear RGB is in
    [0, 1], which compresses L by roughly 100x against textbook CIELab. The deltaL2,
    deltaa2 and deltab2 constants below are tuned to that scale, so the conversion is
    reproduced as-is rather than "fixed". Changing it would invalidate the constants.
    """
    im = np.asarray(im, dtype=np.float64) / 255.0
    mask = im >= 0.04045
    im = np.where(mask, ((im + 0.055) / 1.055) ** 2.4, im / 12.92)
    matrix = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])
    c = im @ matrix.T
    c[..., 0] /= 95.047
    c[..., 1] /= 100.000
    c[..., 2] /= 108.833
    m = c >= 0.008856
    c = np.where(m, np.cbrt(np.abs(c)), 7.787 * c + 16.0 / 116.0)
    lab = np.empty_like(c)
    lab[..., 0] = 116.0 * c[..., 1] - 16.0
    lab[..., 1] = 500.0 * (c[..., 0] - c[..., 1])
    lab[..., 2] = 200.0 * (c[..., 1] - c[..., 2])
    return lab

# ------------------------------------------------------------------ filters

def conv2_same(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MATLAB conv2(x, y, 'same'): the double 180-degree rotation fixes the alignment."""
    return np.rot90(signal.fftconvolve(np.rot90(x, 2), np.rot90(y, 2), mode="same"), 2)


def gauss1d(halfsupport: int, sigma: float, center: float = 0.0) -> np.ndarray:
    t = np.arange(-halfsupport, halfsupport + 1, dtype=np.float64)
    k = np.exp(-((t - center) ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).reshape(1, -1)


def overlapconv(kernel: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Convolution counting only the overlapping part of the kernel (RRoverlapconv)."""
    out = conv2_same(im, kernel)
    overlap = conv2_same(np.ones_like(im), kernel)
    return kernel.sum() * out / overlap


def filt2(kernel: np.ndarray, im: np.ndarray) -> np.ndarray:
    """filt2 with odd reflection at the border (no repeat of the edge pixel)."""
    ky, kx = kernel.shape
    iy, ix = im.shape
    big = np.pad(im, ((ky, ky), (kx, kx)), mode="reflect")
    big = conv2_same(big, kernel)
    return big[ky:ky + iy, kx:kx + ix]


def dog1_filters(a: int, sigma: float):
    sigi, sigo = 0.71 * sigma, 1.14 * sigma
    t = np.arange(-a, a + 1, dtype=np.float64)
    gi = np.exp(-t ** 2 / (2 * sigi ** 2))
    go = np.exp(-t ** 2 / (2 * sigo ** 2))
    return (gi / gi.sum()).reshape(1, -1), (go / go.sum()).reshape(1, -1)

# ------------------------------------------------------------------ gaussian pyramid

BINOM5 = np.array([[1.0, 4.0, 6.0, 4.0, 1.0]]) / 16.0


def gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """Pyramid as in pyrtools: binomial-5 kernel, decimation by two."""
    out, cur = [img.astype(np.float64)], img.astype(np.float64)
    for _ in range(levels - 1):
        t = filt2(BINOM5, cur)
        t = filt2(BINOM5.T, t)
        cur = t[::2, ::2]
        out.append(cur)
    return out


def _reduce(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    k = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]]) if kernel is None else kernel
    t = filt2(k, img)
    t = t[:, ::2]
    t = filt2(k.T, t)
    return t[::2, :]


def _expand(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    k = (np.array([[0.05, 0.25, 0.4, 0.25, 0.05]]) if kernel is None else kernel) * 2
    ys, xs = img.shape
    tmp = np.zeros((ys, 2 * xs))
    tmp[:, ::2] = img
    tmp = overlapconv(k, tmp)
    out = np.zeros((2 * ys, 2 * xs))
    out[::2, :] = tmp
    return overlapconv(k.T, out)


def collapse_max(levels: list) -> np.ndarray:
    """Collapse scales by taking the maximum, upsampling coarser levels first."""
    kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
    k2 = signal.convolve2d(kernel, kernel.T)
    out = levels[0].copy()
    for scale in range(1, len(levels)):
        cur = levels[scale]
        for _ in range(scale, 0, -1):
            ys, xs = cur.shape
            up = np.zeros((2 * ys, 2 * xs))
            up[::2, ::2] = cur
            cur = conv2_same(up, k2 * 4.0)
        h = min(out.shape[0], cur.shape[0])
        w = min(out.shape[1], cur.shape[1])
        out[:h, :w] = np.maximum(out[:h, :w], cur[:h, :w])
    return out

# ------------------------------------------------------------------ Feature Congestion

def colour_clutter(Lp, ap, bp, pool_sigma: float = 3.0) -> list:
    """Local colour variability: volume of the (L, a, b) covariance ellipsoid."""
    dL2, da2, db2 = 0.0007 ** 2, 0.1 ** 2, 0.05 ** 2
    G = gauss1d(int(round(2 * pool_sigma)), pool_sigma)

    def blur(x):
        return overlapconv(G.T, overlapconv(G, x))

    out = []
    for L, a, b in zip(Lp, ap, bp):
        EL, Ea, Eb = blur(L), blur(a), blur(b)
        cLL = blur(L * L) - EL * EL + dL2
        cLa = blur(L * a) - EL * Ea
        cLb = blur(L * b) - EL * Eb
        caa = blur(a * a) - Ea * Ea + da2
        cab = blur(a * b) - Ea * Eb
        cbb = blur(b * b) - Eb * Eb + db2
        det = (cLL * (caa * cbb - cab * cab)
               - cLa * (cLa * cbb - cab * cLb)
               + cLb * (cLa * cab - caa * cLb))
        out.append(np.sqrt(np.abs(det)) ** (1.0 / 3.0))
    return out


def contrast_clutter(Lp, filt_sigma: float = 1.0, pool_sigma: float | None = None) -> list:
    """Local variability of luminance contrast, from a difference-of-Gaussians band."""
    pool_sigma = 3 * filt_sigma if pool_sigma is None else pool_sigma
    gi, go = dog1_filters(int(round(filt_sigma * 3)), filt_sigma)
    contrast = []
    for L in Lp:
        inner = filt2(gi.T, filt2(gi, L))
        outer = filt2(go.T, filt2(go, L))
        contrast.append(np.abs(inner - outer))
    G = gauss1d(int(round(pool_sigma * 2)), pool_sigma)

    def blur(x):
        return overlapconv(G.T, overlapconv(G, x))

    return [np.sqrt(np.abs(blur(c * c) - blur(c) ** 2)) for c in contrast]


def _orient_filters(sigma: float):
    """Second-derivative-of-Gaussian filters at 0, 45, 90 and 135 degrees."""
    hs = int(round(3 * sigma))
    gx = gauss1d(hs, sigma)
    Ga = signal.convolve2d(gx, gauss1d(hs, sigma, sigma).T)
    Ga /= Ga.sum()
    Gb = signal.convolve2d(gx, gauss1d(hs, sigma).T)
    Gb /= Gb.sum()
    Gc = signal.convolve2d(gx, gauss1d(hs, sigma, -sigma).T)
    Gc /= Gc.sum()
    H = -Ga + 2 * Gb - Gc
    V = H.T

    def rot(m, ang):
        r = ndimage.rotate(m, ang, order=3, reshape=False, mode="constant", cval=0.0)
        return r / r.sum()

    R = -rot(Ga, 45) + 2 * rot(Gb, 45) - rot(Gc, 45)
    L = -rot(Ga, -45) + 2 * rot(Gb, -45) - rot(Gc, -45)
    return H, V, L, R


def orientation_clutter(Lp) -> list:
    """Local variability of orientation, via oriented opponent energy."""
    noise_energy, filter_scale, pool_scale = 1.0, 16.0 / 14.0 * 1.75, 1.75
    H, V, Lf, Rf = _orient_filters(filter_scale)
    pool_kernel = gauss1d(int(round(2 * pool_scale)), pool_scale)
    angles = []
    for L in Lp:
        bands = [filt2(H, L), filt2(V, L), filt2(Lf, L), filt2(Rf, L)]
        bands = [b ** 2 for b in bands]
        bands = [_reduce(_expand(b, pool_kernel), pool_kernel) for b in bands]
        hv = bands[0] - bands[1]
        dd = bands[3] - bands[2]
        total = bands[0] + bands[1] + bands[2] + bands[3] + noise_energy
        angles.append((hv / total, dd / total))

    noise, pool_sigma = 0.001, 7.0 / 2.0
    G = gauss1d(int(round(8 * pool_sigma)), 4 * pool_sigma)

    def blur(x):
        return overlapconv(G.T, overlapconv(G, x))

    out = []
    for cmx, smx in angles:
        Dc, Ds = blur(cmx), blur(smx)
        c00 = blur(cmx * cmx) - Dc * Dc + noise
        c01 = blur(cmx * smx) - Dc * Ds
        c11 = blur(smx * smx) - Ds * Ds + noise
        det = c00 * c11 - c01 * c01
        out.append(np.abs(det) ** 0.25)
    return out


def feature_congestion(rgb: np.ndarray, numlevels: int = 3, p: float = 1.0):
    """Feature Congestion: a scalar, a local clutter map, and the three components."""
    lab = rgb_to_lab(rgb)
    Lp = gaussian_pyramid(lab[..., 0], numlevels)
    ap = gaussian_pyramid(lab[..., 1], numlevels)
    bp = gaussian_pyramid(lab[..., 2], numlevels)
    col = collapse_max(colour_clutter(Lp, ap, bp))
    con = collapse_max(contrast_clutter(Lp))
    ori = collapse_max(orientation_clutter(Lp))
    h = min(col.shape[0], con.shape[0], ori.shape[0])
    w = min(col.shape[1], con.shape[1], ori.shape[1])
    cmap = (col[:h, :w] / 0.2088) + (con[:h, :w] / 0.0660) + (ori[:h, :w] / 0.0269)
    scalar = float(np.mean(cmap ** p) ** (1.0 / p))
    parts = {"colour": float(np.mean(col)),
             "contrast": float(np.mean(con)),
             "orientation": float(np.mean(ori))}
    return scalar, cmap, parts

# ------------------------------------------------------------------ steerable pyramid

def _rcos_fn(width=1.0, position=0.0, values=(0.0, 1.0)):
    sz = 256
    X = np.pi * np.arange(-sz - 1, 2) / (2 * sz)
    Y = values[0] + (values[1] - values[0]) * np.cos(X) ** 2
    Y[0] = Y[1]
    Y[sz + 2] = Y[sz + 1]
    X = position + (2 * width / np.pi) * (X + np.pi / 4)
    return X, Y


def _point_op(im, Y, X):
    return np.interp(im.ravel(), X, Y).reshape(im.shape)


def _polar(dims):
    ctr = np.ceil((np.array(dims) + 0.5) / 2).astype(int)
    xr = np.linspace(-1, 1, dims[1] + 1)[:-1]
    yr = np.linspace(-1, 1, dims[0] + 1)[:-1]
    xramp, yramp = np.meshgrid(xr, yr)
    angle = np.arctan2(yramp, xramp)
    rad = np.sqrt(xramp ** 2 + yramp ** 2)
    rad[ctr[0] - 1, ctr[1] - 1] = rad[ctr[0] - 1, ctr[1] - 2]
    return angle, np.log2(rad)


def _angular_lut(order: int, nbands: int):
    lutsize = 1024
    Xcosn = np.pi * np.arange(-(2 * lutsize + 1), lutsize + 2) / lutsize
    const = (2 ** (2 * order)) * (math.factorial(order) ** 2) / (nbands * math.factorial(2 * order))
    alfa = ((np.pi + Xcosn) % (2 * np.pi)) - np.pi
    Ycosn = 2 * np.sqrt(const) * (np.cos(Xcosn) ** order) * (np.abs(alfa) < np.pi / 2)
    return Xcosn, Ycosn


def tiling_error(dims=(64, 64), height: int = 3, order: int = 3) -> np.ndarray:
    """Self-check on the mask construction: the squared transfer functions must sum to 1.

    Band (i, b) has transfer lo0 * (product of lomask up to i) * himask_i * anglemask_b;
    the highpass residual has hi0, the lowpass residual lo0 * (product of all lomask).

    The angular masks cover only half the frequency plane, the other half being supplied
    by the conjugate symmetry of a real image's spectrum, which is why Ycosn carries the
    factor 2. The identity is therefore checked on angles paired as (theta, theta + pi).
    Computed at full resolution, since decimation changes the cost, not the transfers.
    """
    nbands = order + 1
    angle, log_rad = _polar(dims)
    Xrcos, Yrcos = _rcos_fn(1.0, -0.5, (0.0, 1.0))
    Yrcos = np.sqrt(Yrcos)
    YIrcos = np.sqrt(np.clip(1.0 - Yrcos ** 2, 0.0, None))

    total = _point_op(log_rad, Yrcos, Xrcos) ** 2
    lo_cum = _point_op(log_rad, YIrcos, Xrcos) ** 2
    Xcosn, Ycosn = _angular_lut(order, nbands)

    for _ in range(height):
        Xrcos = Xrcos - np.log2(2.0)
        himask = _point_op(log_rad, Yrcos, Xrcos)
        angle_flip = ((angle + 2 * np.pi) % (2 * np.pi)) - np.pi
        ang_sum = np.zeros_like(log_rad)
        for b in range(nbands):
            Xa = Xcosn[0] + np.pi * b / nbands + (Xcosn[1] - Xcosn[0]) * np.arange(len(Ycosn))
            ang_sum += _point_op(angle, Ycosn, Xa) ** 2
            ang_sum += _point_op(angle_flip, Ycosn, Xa) ** 2
        total = total + lo_cum * (himask ** 2) * ang_sum / 4.0
        YIrcos = np.abs(np.sqrt(np.clip(1.0 - Yrcos ** 2, 0.0, None)))
        lo_cum = lo_cum * _point_op(log_rad, YIrcos, Xrcos) ** 2
    return total + lo_cum


def steerable_bands(img: np.ndarray, height: int = 3, order: int = 3) -> list:
    """Steerable pyramid in the frequency domain (equivalent to SteerablePyramidFreq).

    Returns the highpass residual, height x (order + 1) oriented bands, lowpass residual.
    """
    nbands = order + 1
    angle, log_rad = _polar(img.shape)
    Xrcos, Yrcos = _rcos_fn(1.0, -0.5, (0.0, 1.0))
    Yrcos = np.sqrt(Yrcos)
    YIrcos = np.sqrt(np.clip(1.0 - Yrcos ** 2, 0.0, None))

    dft = np.fft.fftshift(np.fft.fft2(img))
    bands = [np.real(np.fft.ifft2(np.fft.ifftshift(dft * _point_op(log_rad, Yrcos, Xrcos))))]
    lodft = dft * _point_op(log_rad, YIrcos, Xrcos)
    Xcosn, Ycosn = _angular_lut(order, nbands)

    for _ in range(height):
        Xrcos = Xrcos - np.log2(2.0)
        himask = _point_op(log_rad, Yrcos, Xrcos)
        for b in range(nbands):
            Xa = Xcosn[0] + np.pi * b / nbands + (Xcosn[1] - Xcosn[0]) * np.arange(len(Ycosn))
            anglemask = _point_op(angle, Ycosn, Xa)
            banddft = ((-1j) ** order) * lodft * anglemask * himask
            bands.append(np.real(np.fft.ifft2(np.fft.ifftshift(banddft))))
        dims = np.array(lodft.shape)
        ctr = np.ceil((dims + 0.5) / 2).astype(int)
        lodims = np.ceil((dims - 0.5) / 2).astype(int)
        loctr = np.ceil((lodims + 0.5) / 2).astype(int)
        lo, hi = ctr - loctr, ctr - loctr + lodims
        log_rad = log_rad[lo[0]:hi[0], lo[1]:hi[1]]
        angle = angle[lo[0]:hi[0], lo[1]:hi[1]]
        lodft = lodft[lo[0]:hi[0], lo[1]:hi[1]]
        YIrcos = np.abs(np.sqrt(np.clip(1.0 - Yrcos ** 2, 0.0, None)))
        lodft = lodft * _point_op(log_rad, YIrcos, Xrcos)

    bands.append(np.real(np.fft.ifft2(np.fft.ifftshift(lodft))))
    return bands


def _entropy(x: np.ndarray) -> float:
    """Shannon entropy with the reference port's binning: ceil(sqrt(N)) uniform bins."""
    x = x.ravel()
    n = x.shape[0]
    nbins = int(np.ceil(np.sqrt(n)))
    if nbins <= 1:
        return 0.0
    edges = np.histogram(x, bins=nbins - 1)[1]
    idx = np.digitize(x, edges)
    hist = np.bincount(np.clip(idx - 1, 0, len(edges) - 1), minlength=len(edges)).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    hist = hist / total
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist)))


def subband_entropy(rgb: np.ndarray, wlevels: int = 3, wght_chrom: float = 0.0625) -> float:
    """Subband Entropy: bits needed to encode the image, luminance plus weighted chroma."""
    lab = rgb_to_lab(rgb)
    order = 3  # four orientations
    se = float(np.mean([_entropy(b) for b in steerable_bands(lab[..., 0], wlevels, order)]))
    for ch in (1, 2):
        se += wght_chrom * float(np.mean([_entropy(b) for b in steerable_bands(lab[..., ch], wlevels, order)]))
    return se / (1.0 + 2.0 * wght_chrom)

# ------------------------------------------------------------------ extras

def edge_density(rgb: np.ndarray) -> float:
    """Crude clutter proxy: share of pixels on a strong edge."""
    g = np.asarray(Image.fromarray(rgb.astype(np.uint8)).convert("L"), dtype=np.float64)
    mag = np.hypot(ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0))
    return float((mag > 0.1 * mag.max()).mean()) if mag.max() > 0 else 0.0


def colour_mass(rgb: np.ndarray, blur_px: float = 2.5) -> float:
    """What survives peripheral vision: mean Lab chroma of a blurred frame.

    Dark text contributes almost nothing, a solid colour fill contributes a lot. Use this
    to rank what pulls the eye first, not how busy the frame is overall.
    """
    from PIL import ImageFilter
    im = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_px))
    lab = rgb_to_lab(np.asarray(im))
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())

# ------------------------------------------------------------------ driver

def load_rgb(path: str, width: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if width and im.width != width:
        h = max(1, int(round(im.height * width / im.width)))
        im = im.resize((width, h), Image.LANCZOS)
    return np.asarray(im)


def measure(path: str, width: int | None = None, maps_dir: str | None = None) -> dict:
    rgb = load_rgb(path, width)
    fc, cmap, parts = feature_congestion(rgb)
    res = {
        "file": os.path.basename(path),
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "feature_congestion": round(fc, 4),
        "subband_entropy": round(subband_entropy(rgb), 4),
        "edge_density": round(edge_density(rgb), 4),
        "colour_mass": round(colour_mass(rgb), 4),
        "components": {k: round(v, 6) for k, v in parts.items()},
    }
    if maps_dir:
        os.makedirs(maps_dir, exist_ok=True)
        norm = (cmap - cmap.min()) / max(cmap.max() - cmap.min(), 1e-9)
        out = os.path.join(maps_dir, os.path.splitext(os.path.basename(path))[0] + "-fc.png")
        Image.fromarray((norm * 255).astype(np.uint8)).save(out)
        res["map"] = out
    return res


def selftest() -> int:
    print("1. Steerable pyramid tiles the frequency plane (squared transfers sum to 1)")
    total = tiling_error((64, 64), height=3, order=3)
    dev = np.abs(total - 1.0)
    ok_tiling = dev.mean() < 0.01 and np.median(dev) < 0.005
    print(f"   mean {total.mean():.6f}, deviation from 1: mean {dev.mean():.2e}, "
          f"median {np.median(dev):.2e}, max {dev.max():.2e}")
    print(f"   tiling correct: {ok_tiling}")

    print("2. Clutter rises with the number of items on the frame")
    rng = np.random.default_rng(1)
    rows = []
    for count in (0, 8, 40, 200):
        canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
        for _ in range(count):
            y, x = rng.integers(0, 236, 2)
            canvas[y:y + 18, x:x + 18] = rng.integers(0, 256, 3)
        fc, _, _ = feature_congestion(canvas)
        se = subband_entropy(canvas)
        rows.append((count, fc, se))
        print(f"   items {count:3d}: FC {fc:8.3f}   SE {se:6.3f}")
    fc_ok = all(rows[i][1] < rows[i + 1][1] for i in range(len(rows) - 1))
    se_ok = all(rows[i][2] < rows[i + 1][2] for i in range(len(rows) - 1))
    print(f"   FC monotonic: {fc_ok}; SE monotonic: {se_ok}")

    print("3. Blank frame against colour noise")
    white = np.full((128, 128, 3), 255, dtype=np.uint8)
    noise = rng.integers(0, 256, (128, 128, 3)).astype(np.uint8)
    fw, _, _ = feature_congestion(white)
    fn, _, _ = feature_congestion(noise)
    print(f"   FC blank {fw:.3f} against noise {fn:.3f}; "
          f"SE {subband_entropy(white):.3f} against {subband_entropy(noise):.3f}")
    ok = ok_tiling and fc_ok and se_ok and fw < fn
    print(f"\nself-test {'passed' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Rosenholtz visual clutter measures")
    ap.add_argument("images", nargs="*", help="frames to measure")
    ap.add_argument("--width", type=int, default=None,
                    help="rescale to the actual display width before measuring")
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--maps", default=None, help="directory for local clutter maps")
    ap.add_argument("--selftest", action="store_true", help="verify the implementation")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.images:
        ap.error("pass frames to measure, or --selftest")

    out = []
    for path in args.images:
        r = measure(path, args.width, args.maps)
        out.append(r)
        print(f"{r['file']}: FC {r['feature_congestion']}, SE {r['subband_entropy']}, "
              f"edges {r['edge_density']}, colour {r['colour_mass']}, "
              f"size {r['size'][0]}x{r['size'][1]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
