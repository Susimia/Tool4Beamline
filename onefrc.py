#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
onefrc.py — single-image FRC with 1/7, 1/2-bit, 1-bit threshold AND PSD-SSNR

Run (examples):
  pip install numpy matplotlib tifffile         # + scipy if you use --auto
  python onefrc.py *.tif --px 5.2 --criterion all
  # With robust preprocessing on float/processed images:
  python onefrc.py *.tif --px 5.2 --criterion all --auto --apodize --rmax 0.48 --nsplits 40
  # With a separate noise reference for PSD:
  python onefrc.py *.tif --noise-tiff dark.tif --criterion all

Outputs:
  <stem>_1frc.png  (locked aesthetics; FRC plot)
  <stem>_1frc.csv  (FRC, thresholds, PSD power/noise/SSNR, SSNR thresholds)
"""

import argparse
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from numpy.fft import fft2
from pathlib import Path

# --- Aesthetics: lock figure style ---
plt.rcParams.update({
    "figure.figsize": (6, 4.2),
    "savefig.dpi": 200,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})

# --- Core ---
def radial_bins(shape, rmax=0.5):
    ny, nx = shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    r = np.hypot(fx, fy)
    dr = 1.0 / min(ny, nx)
    nbins = int(np.floor(rmax / dr)) + 1
    edges = np.linspace(0.0, rmax, nbins + 1)
    mask = r < rmax  # strict to avoid off-by-one
    bin_idx = np.digitize(r, edges) - 1
    bin_idx[(bin_idx < 0) | (bin_idx >= nbins)] = -1
    return bin_idx, mask, edges

def _ring_bincount(bin_idx, arr, nbins):
    valid = bin_idx >= 0
    bins = bin_idx[valid]
    vals = arr[valid]
    return np.bincount(bins.ravel(), weights=vals.ravel(), minlength=nbins).astype(np.float64)[:nbins]

def frc_curve(img1, img2, bin_idx, mask, nbins):
    F1 = fft2(img1)
    F2 = fft2(img2)
    cross = np.real(F1 * np.conj(F2))
    p1 = np.abs(F1) ** 2
    p2 = np.abs(F2) ** 2
    num  = _ring_bincount(bin_idx, cross, nbins)
    den1 = _ring_bincount(bin_idx, p1, nbins)
    den2 = _ring_bincount(bin_idx, p2, nbins)
    counts = np.bincount((bin_idx[bin_idx >= 0]).ravel(), minlength=nbins)[:nbins]
    frc = num / (np.sqrt(den1 * den2) + 1e-20)
    frc[counts == 0] = np.nan
    return frc, counts

def psd_power(img, bin_idx, nbins):
    F = fft2(img)
    P = np.abs(F) ** 2
    counts = np.bincount((bin_idx[bin_idx >= 0]).ravel(), minlength=nbins)[:nbins]
    power = _ring_bincount(bin_idx, P, nbins)
    power[counts == 0] = np.nan
    return power, counts

def threshold_crossing_const(f, y, tau):
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        if np.any(np.isnan([y0, y1])): 
            continue
        if (y0 >= tau) and (y1 < tau):
            t = (tau - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.0
            return float(f[i] + t * (f[i + 1] - f[i]))
    return float("nan")

def threshold_crossing_curve(f, y, tcurve):
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        t0, t1 = tcurve[i], tcurve[i + 1]
        if np.any(np.isnan([y0, y1, t0, t1])): 
            continue
        if (y0 >= t0) and (y1 < t1):
            dy0 = y0 - t0; dy1 = y1 - t1
            t = dy0 / (dy0 - dy1) if (dy0 - dy1) != 0 else 0.0
            return float(f[i] + t * (f[i + 1] - f[i]))
    return float("nan")

# Bit-based FRC thresholds (van Heel & Schatz)
def halfbit_threshold(counts):
    # T_1/2-bit(i) = (0.2071 + 1.9102/sqrt(n_i)) / (1.2071 + 0.9102/sqrt(n_i))
    n = np.maximum(counts.astype(np.float64), 1.0)
    s = 1.0 / np.sqrt(n)
    return (0.2071 + 1.9102 * s) / (1.2071 + 0.9102 * s)

def onebit_threshold(counts):
    # T_1-bit(i)   = (0.5 + 2.4142/sqrt(n_i)) / (1.5 + 1.4142/sqrt(n_i))
    n = np.maximum(counts.astype(np.float64), 1.0)
    s = 1.0 / np.sqrt(n)
    return (0.5 + 2.4142 * s) / (1.5 + 1.4142 * s)

# Convert FRC threshold T to equivalent SSNR threshold: SSNR = T/(1-T)
def to_ssnr_threshold(t):
    return t / np.maximum(1.0 - t, 1e-12)

# --- Optional preprocessing (OFF by default to keep original values) ---
def preprocess(img, auto=False, apodize=False):
    arr = np.asarray(img, dtype=np.float64)
    arr = np.nan_to_num(arr, copy=False)
    arr[arr < 0] = 0.0
    note = "original values"
    if auto:
        from scipy.ndimage import gaussian_filter  # lazy import
        p99 = np.percentile(arr, 99.0)
        scale = 100.0 / p99 if p99 > 0 else 1.0
        arr = np.clip(arr * scale, 0, None)
        bg = gaussian_filter(arr, sigma=5.0)
        bg[bg <= 0] = 1.0
        arr = arr / bg
        p99b = np.percentile(arr, 99.0)
        scale2 = 100.0 / p99b if p99b > 0 else 1.0
        arr *= scale2
        note = f"auto norm (x{scale:.3g}); flatten; rescale x{scale2:.3g}"
    if apodize:
        def tukey_window(n, alpha=0.25):
            if n <= 1: return np.ones(n)
            x = np.linspace(0, 1, n); w = np.ones(n); edge = alpha/2
            i1 = x < edge; i2 = x > 1 - edge
            w[i1] = 0.5*(1 + np.cos(np.pi*(2*x[i1]/alpha - 1)))
            w[i2] = 0.5*(1 + np.cos(np.pi*(2*(1 - x[i2])/alpha - 1)))
            return w
        wy = tukey_window(arr.shape[0], 0.25)[:, None]
        wx = tukey_window(arr.shape[1], 0.25)[None, :]
        arr *= (wy * wx)
        note += "; apodized"
    return arr, note

# --- Main 1FRC + PSD ---
def run_analysis(img, split="poisson", nsplits=20, seed=0, auto=False, apodize=False,
                 rmax=0.5, noise_img=None, psd_tail_frac=0.10):
    # Preprocess primary image
    counts_img, note = preprocess(img, auto=auto, apodize=apodize)
    bin_idx, mask, edges = radial_bins(counts_img.shape, rmax)
    nbins = len(edges) - 1
    freq = 0.5 * (edges[:-1] + edges[1:])
    ring_counts = np.bincount((bin_idx[bin_idx >= 0]).ravel(), minlength=nbins)[:nbins]

    # --- 1FRC ---
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(nsplits):
        if split == "binomial":
            n = np.rint(counts_img).astype(np.int64)
            a = rng.binomial(n, 0.5).astype(np.float64)
            b = (n - a).astype(np.float64)
        else:
            a = rng.poisson(0.5 * counts_img).astype(np.float64)
            b = rng.poisson(0.5 * counts_img).astype(np.float64)
        frc, _ = frc_curve(a, b, bin_idx, mask, nbins)
        curves.append(frc)
    curves = np.vstack(curves)
    frc_mean = np.nanmean(curves, axis=0)
    s = np.nanstd(curves, axis=0, ddof=1)
    n = np.sum(~np.isnan(curves), axis=0)
    se = np.where(n > 0, s / np.sqrt(n), np.nan)
    z = 1.96
    frc_lo, frc_hi = frc_mean - z * se, frc_mean + z * se

    # --- PSD / SSNR ---
    psd_power_vals, _ = psd_power(counts_img, bin_idx, nbins)

    if noise_img is not None:
        # Use provided noise reference (no auto-normalization; optional apodize for same window)
        noise_counts, _ = preprocess(noise_img, auto=False, apodize=apodize)
        psd_noise_vals, _ = psd_power(noise_counts, bin_idx, nbins)
    else:
        # Flat noise estimate from the high-frequency tail (robust median)
        tail_bins = max(5, int(np.ceil(nbins * psd_tail_frac)))
        tail = psd_power_vals[-tail_bins:]
        flat = np.nanmedian(tail[np.isfinite(tail)]) if np.any(np.isfinite(tail)) else np.nan
        psd_noise_vals = np.full_like(psd_power_vals, flat, dtype=np.float64)

    eps = 1e-20
    psd_ssnr = (psd_power_vals - psd_noise_vals) / (np.maximum(psd_noise_vals, eps))

    # Threshold curves
    th_half = halfbit_threshold(ring_counts)
    th_one  = onebit_threshold(ring_counts)
    # SSNR thresholds (equivalents)
    tau_17 = 1.0 / 7.0
    ssnr_thr_17 = tau_17 / (1.0 - tau_17)        # = 1/6
    ssnr_thr_half = to_ssnr_threshold(th_half)
    ssnr_thr_one  = to_ssnr_threshold(th_one)

    return {
        "freq": freq,
        "frc_mean": frc_mean, "frc_lo": frc_lo, "frc_hi": frc_hi,
        "ring_counts": ring_counts,
        "th_half": th_half, "th_one": th_one,
        "psd_power": psd_power_vals, "psd_noise": psd_noise_vals, "psd_ssnr": psd_ssnr,
        "ssnr_thr_17": np.full_like(freq, ssnr_thr_17),
        "ssnr_thr_half": ssnr_thr_half, "ssnr_thr_one": ssnr_thr_one,
        "note": note,
    }


# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Single-image FRC with 1/7, 1/2-bit, 1-bit, and PSD-SSNR.")
    ap.add_argument("tiff", help="Input TIFF (single frame; first plane used if stack).")
    ap.add_argument("--px", type=float, default=None, help="Pixel size (real units per pixel).")
    ap.add_argument("--split", choices=["poisson", "binomial"], default="poisson",
                    help="Random split: poisson (default) or binomial (integer counts).")
    ap.add_argument("--nsplits", type=int, default=20, help="Number of random splits for CI.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--auto", action="store_true", help="Enable robust normalization/flattening.")
    ap.add_argument("--apodize", action="store_true", help="Apply a Tukey window.")
    ap.add_argument("--rmax", type=float, default=0.5,
                    help="Max radius in cycles/pixel (e.g., 0.48 to avoid Nyquist artifacts).")
    ap.add_argument("--criterion", choices=["1over7","halfbit","onebit","both","all"], default="1over7",
                    help="Which thresholds to annotate on the plot (FRC domain).")
    ap.add_argument("--noise-tiff", type=str, default=None,
                    help="Optional noise/blank TIFF for PSD noise estimation (otherwise use tail median).")
    ap.add_argument("--psd-tail-frac", type=float, default=0.10,
                    help="Fraction of highest frequencies to estimate flat PSD noise (default 0.10).")
    ap.add_argument("--show-psd", action="store_true",
                    help="Overlay PSD-SSNR on a secondary axis (off by default).")
    args = ap.parse_args()

    # Load main image
    arr = tiff.imread(args.tiff)
    if arr.ndim > 2:
        arr = arr[0]

    # Optional noise reference
    noise_img = None
    if args.noise_tiff:
        noise_img = tiff.imread(args.noise_tiff)
        if noise_img.ndim > 2:
            noise_img = noise_img[0]

    out = run_analysis(
        arr, split=args.split, nsplits=args.nsplits, seed=args.seed,
        auto=args.auto, apodize=args.apodize, rmax=args.rmax,
        noise_img=noise_img, psd_tail_frac=args.psd_tail_frac
    )

    freq = out["freq"]
    m, lo, hi = out["frc_mean"], out["frc_lo"], out["frc_hi"]
    th_half, th_one = out["th_half"], out["th_one"]
    psd_power, psd_noise, psd_ssnr = out["psd_power"], out["psd_noise"], out["psd_ssnr"]
    ssnr_thr_17, ssnr_thr_half, ssnr_thr_one = out["ssnr_thr_17"], out["ssnr_thr_half"], out["ssnr_thr_one"]

    # FRC cutoffs
    fcut_17   = threshold_crossing_const(freq, m, tau=1/7) if args.criterion in ("1over7","both","all") else np.nan
    fcut_half = threshold_crossing_curve(freq, m, th_half)  if args.criterion in ("halfbit","both","all") else np.nan
    fcut_one  = threshold_crossing_curve(freq, m, th_one)   if args.criterion in ("onebit","all") else np.nan

    # PSD-SSNR cutoffs (match same criteria in SSNR domain)
    def crossing_ssnr_const(ssnr, thr_const):
        return threshold_crossing_const(freq, ssnr, thr_const)
    def crossing_ssnr_curve(ssnr, thr_curve):
        return threshold_crossing_curve(freq, ssnr, thr_curve)

    psdcut_17   = crossing_ssnr_const(psd_ssnr, ssnr_thr_17[0]) if args.criterion in ("1over7","both","all") else np.nan
    psdcut_half = crossing_ssnr_curve(psd_ssnr, ssnr_thr_half)   if args.criterion in ("halfbit","both","all") else np.nan
    psdcut_one  = crossing_ssnr_curve(psd_ssnr, ssnr_thr_one)    if args.criterion in ("onebit","all") else np.nan

    # Resolutions
    def res_pair(fcut):
        px = (1.0 / fcut) if (np.isfinite(fcut) and fcut > 0) else np.nan
        ru = (args.px / fcut) if (args.px is not None and np.isfinite(fcut) and fcut > 0) else None
        return px, ru

    res_px_17,  res_units_17    = res_pair(fcut_17)
    res_px_half, res_units_half = res_pair(fcut_half)
    res_px_one,  res_units_one  = res_pair(fcut_one)

    psd_res_px_17,  psd_res_units_17    = res_pair(psdcut_17)
    psd_res_px_half, psd_res_units_half = res_pair(psdcut_half)
    psd_res_px_one,  psd_res_units_one  = res_pair(psdcut_one)

    # --- Plot (FRC main; optional PSD overlay) ---
    stem = Path(args.tiff).with_suffix("")
    png = f"{stem}_1frc.png"
    csv = f"{stem}_1frc.csv"

    lw = 1.8
    dash    = (0, (6, 3))   # dashed for 1/7 line
    dot     = (0, (1, 3))   # dotted for cutoff markers
    dashdot = (0, (3, 3))   # dash-dot for 1/2-bit curve
    longdash= (0, (9, 3))   # long dash for 1-bit curve

    fig, ax1 = plt.subplots()
    ax1.plot(freq, m, label="1FRC", lw=lw)
    ax1.fill_between(freq, lo, hi, alpha=0.25, label="95% CI", linewidth=0)

    if args.criterion in ("1over7","both","all"):
        ax1.axhline(1/7, linestyle=dash, lw=1.5, label="1/7 threshold")
        if np.isfinite(fcut_17) and fcut_17 > 0:
            ax1.axvline(fcut_17, linestyle=dot, lw=1.5, label="cutoff (1/7)")

    if args.criterion in ("halfbit","both","all"):
        ax1.plot(freq, th_half, linestyle=dashdot, lw=1.2, label="1/2-bit threshold")
        if np.isfinite(fcut_half) and fcut_half > 0:
            ax1.axvline(fcut_half, linestyle=dot, lw=1.5, label="cutoff (1/2-bit)")

    if args.criterion in ("onebit","all"):
        ax1.plot(freq, th_one, linestyle=longdash, lw=1.2, label="1-bit threshold")
        if np.isfinite(fcut_one) and fcut_one > 0:
            ax1.axvline(fcut_one, linestyle=dot, lw=1.5, label="cutoff (1-bit)")

    ax1.set_xlabel("Spatial frequency (cycles/pixel)")
    ax1.set_ylabel("FRC")
    ax1.set_ylim(0, 1.05); ax1.set_xlim(0, args.rmax)

    if args.show_psd:
        ax2 = ax1.twinx()
        ax2.plot(freq, psd_ssnr, lw=1.2, alpha=0.85, label="PSD-SSNR")
        # Optional: show SSNR thresholds as thin lines (comment out to keep super clean)
        # ax2.axhline(ssnr_thr_17[0], linestyle=dash, lw=1.0)
        ax2.set_ylabel("SSNR (PSD)")
        # Add legend entries from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
    else:
        ax1.legend()

    fig.tight_layout()
    fig.savefig(png, dpi=200)
    plt.close(fig)

    # --- CSV: include FRC thresholds, PSD power/noise/SSNR + SSNR thresholds ---
    with open(csv, "w") as f:
        f.write("freq_cyc_per_px,frc_mean,frc_lo,frc_hi,halfbit_threshold,onebit_threshold,")
        f.write("psd_power,psd_noise,psd_ssnr,oneover7_ssnr_threshold,halfbit_ssnr_threshold,onebit_ssnr_threshold\n")
        for fi, mi, loi, hii, thi, t1, P, N, S, S17, Sh, S1 in zip(
            freq, m, lo, hi, th_half, th_one, psd_power, psd_noise, psd_ssnr,
            out["ssnr_thr_17"], out["ssnr_thr_half"], out["ssnr_thr_one"]
        ):
            f.write(f"{fi},{mi},{loi},{hii},{thi},{t1},{P},{N},{S},{S17},{Sh},{S1}\n")

    # --- Console summary ---
    print(f"Mode: {args.split} | {out['note']}")
    if args.criterion in ("1over7","both","all"):
        print(f"[FRC 1/7]     cutoff: {fcut_17:.6f} cyc/px | res: {res_px_17:.3f} px" +
              (f" | {res_units_17:.3f} (units of --px)" if res_units_17 is not None else ""))
        print(f"[PSD 1/7]     cutoff: {psdcut_17:.6f} cyc/px | res: {psd_res_px_17:.3f} px" +
              (f" | {psd_res_units_17:.3f} (units of --px)" if psd_res_units_17 is not None else ""))
    if args.criterion in ("halfbit","both","all"):
        print(f"[FRC 1/2-bit] cutoff: {fcut_half:.6f} cyc/px | res: {res_px_half:.3f} px" +
              (f" | {res_units_half:.3f} (units of --px)" if res_units_half is not None else ""))
        print(f"[PSD 1/2-bit] cutoff: {psdcut_half:.6f} cyc/px | res: {psd_res_px_half:.3f} px" +
              (f" | {psd_res_units_half:.3f} (units of --px)" if psd_res_units_half is not None else ""))
    if args.criterion in ("onebit","all"):
        print(f"[FRC 1-bit]   cutoff: {fcut_one:.6f} cyc/px | res: {res_px_one:.3f} px" +
              (f" | {res_units_one/2:.3f} (units of --px)" if res_units_one is not None else ""))
        print(f"[PSD 1-bit]   cutoff: {psdcut_one:.6f} cyc/px | res: {psd_res_px_one:.3f} px" +
              (f" | {psd_res_units_one/2:.3f} (units of --px)" if psd_res_units_one is not None else ""))
    print(f"Saved: {png}\nSaved: {csv}")

if __name__ == "__main__":
    main()
