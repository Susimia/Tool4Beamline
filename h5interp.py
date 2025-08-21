#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STXM HDF5 -> (PNG image, HDF5 result) converter

Run (examples):
  pip install numpy scipy h5py matplotlib
  python h5interp.py --glob "*.h5" --lines even --use-ref --thres 0.96 --spotlength 30

Outputs:
  <stem>interp.hdf5  (locked aesthetics; FRC plot)
  <stem>intp.png  (a .png figure for view)
  <stem>intp.tiff  (a .tiff figure for view)
"""

import argparse
import glob
import os
import math
import numpy as np
import h5py
import tifffile
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# -------------------------------
# Utilities
# -------------------------------

def as_scalar(x, name="value"):
    """Safe scalar extraction (fixes NumPy 1.25+ deprecation)."""
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return float(np.ravel(arr)[0])

def get_dataset_any(grp, candidates):
    """Return the first existing dataset among candidate names."""
    for name in candidates:
        if name in grp:
            return grp[name][()]
    raise KeyError(f"None of these datasets found: {candidates}")

def save_png_matlab_style(img, out_png, dpi=300, clip=(1, 99), colorbar=False, show_axes=True):
    """Save PNG mimicking MATLAB imagesc + axis image + colormap(gray)."""
    if clip is None:
        vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
    else:
        lo, hi = np.nanpercentile(img, clip)
        vmin, vmax = float(lo), float(hi)

    rc = {
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    H, W = img.shape
    extent = [1, W, H, 1]  # MATLAB-like coordinates

    with plt.rc_context(rc):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest",
                       extent=extent, aspect="equal")
        if show_axes:
            ax.tick_params(direction="in", length=3, width=0.6)
            for s in ax.spines.values():
                s.set_visible(True)
        else:
            ax.axis("off")
        if colorbar:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(direction="in", length=3, width=0.6)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

def center_place(dst_shape, src, fill):
    """Center-place src into a new array of shape dst_shape filled with fill (crop if needed)."""
    H, W = dst_shape
    dst = np.full((H, W), fill, dtype=np.float32)
    h, w = src.shape
    def place_1d(Ldst, Lsrc):
        if Lsrc <= Ldst:
            d0 = (Ldst - Lsrc) // 2
            return (slice(d0, d0 + Lsrc), slice(0, Lsrc))
        else:
            s0 = (Lsrc - Ldst) // 2
            return (slice(0, Ldst), slice(s0, s0 + Ldst))
    (yd, ys) = place_1d(H, h)
    (xd, xs) = place_1d(W, w)
    dst[yd, xd] = src[ys, xs]
    return dst


# Reference mask 
def refmask_find(dset_ref, thres_percent=0.96, spotlength=30, njetspot0=50):
    """
    Port of MATLAB refmask_find with vectorization.
    Returns: refmask (bool 2D) where True=keep, False=mask out.
    """
    ref = np.asarray(dset_ref, dtype=np.float32)
    nxref, nyref = ref.shape
    flat = ref.reshape(-1, order="F")  # MATLAB column-major
    nnxy = flat.size

    # Robust background from middle 50% of sorted values
    s = np.sort(flat)[::-1]
    i0 = math.ceil(0.25 * s.size); i1 = math.ceil(0.75 * s.size)
    bgv = s[i0:i1].mean() if i1 > i0 else s.mean()

    thres = thres_percent * bgv
    njetspot = math.ceil(nnxy / 40200.0 * njetspot0)

    # Below threshold (jets) or above (hot-spots)
    if thres_percent < 1:
        cond = flat < thres
        order_idx = np.argsort(flat)          # small -> large
    else:
        cond = flat > thres
        order_idx = np.argsort(-flat)         # large -> small

    cand = order_idx[cond[order_idx]]
    if cand.size == 0:
        return np.ones((nyref, nxref), dtype=bool)

    # Pick local extrema within ±30 neighborhood
    taken = []
    for j in cand:
        if len(taken) >= njetspot:
            break
        L = max(j - 30, 0); R = min(j + 30, nnxy - 1)
        neigh = flat[L:R+1]
        if (thres_percent < 1 and flat[j] == np.min(neigh)) or \
           (thres_percent >= 1 and flat[j] == np.max(neigh)):
            taken.append(j)

    if not taken:
        return np.ones((nyref, nxref), dtype=bool)

    # Expand each detected spot to fixed length
    halfspot = 15
    flag = cond.astype(np.uint8)
    keep = np.ones(nnxy, dtype=bool)
    for j in taken:
        L = max(j - halfspot, 0)
        seg = flag[L:j]
        nleft = int(seg.sum())
        a = max(j - nleft, 0)
        b = min(j + spotlength - 1, nnxy - 1)
        keep[a:b+1] = False

    refmask = keep.reshape((nxref, nyref), order="F").T  # True=keep
    return refmask


# Core per-file processing
def process_file(path, args):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    out_h5 = f"{stem}interp.hdf5"
    out_png = f"{stem}intp.png"
    out_tiff = f"{stem}intp.tiff"

    if (not args.overwrite) and os.path.exists(out_h5):
        print(f"[Skip] {base} -> {out_h5} exists.")
        return

    with h5py.File(path, "r") as f:
        grp = f.get("FPGA control board")
        if grp is None:
            print(f"[Skip] {base}: missing '/FPGA control board'.")
            return

        # --- read datasets ---
        dset1 = np.asarray(grp["PMT counter"][()], dtype=np.float32)  # (ny, nx)
        pos1  = np.asarray(get_dataset_any(grp, ["positon 1 data", "position 1 data"]), dtype=np.float32)
        pos2  = np.asarray(get_dataset_any(grp, ["positon 2 data", "position 2 data"]), dtype=np.float32)
        rng_um = np.asarray(grp["range(um)"][()], dtype=np.float32)
        stepsize_um = np.asarray(grp["step size(um)"][()], dtype=np.float32)
        energy_ev = as_scalar(grp["energy(eV)"][()], "energy(eV)")

        # If second axis swing too small, try "positon 3 data"
        if (pos2.max() - pos2.min()) < (rng_um[1] / 3.0) and ("positon 3 data" in grp):
            pos2 = np.asarray(grp["positon 3 data"][()], dtype=np.float32)

        # --- units & orientation (match MATLAB) ---
        stepsize_nm = stepsize_um * 1000.0
        stepsizey = float(stepsize_nm[0])  # fast axis in nm
        stepsizex = float(stepsize_nm[1])  # slow axis in nm

        dset1 = dset1.T                    # MATLAB: dset1 = dset1'
        nx, ny = dset1.shape

        # Positions to nm, transpose to match, invert slow axis like MATLAB
        pos_y = (pos1.T * 1000.0).astype(np.float32)
        pos_x = (pos2.T * 1000.0).astype(np.float32)
        pos_x = float(pos_x.max()) - pos_x

        # Normalize to 1-based step indices
        xmin, xmax = float(pos_y.min()), float(pos_y.max())
        ymin, ymax = float(pos_x.min()), float(pos_x.max())
        pos_y = (pos_y - xmin + stepsizey) / stepsizey
        pos_x = (pos_x - ymin + stepsizex) / stepsizex
        nxd = int(math.ceil((xmax - xmin) / stepsizey) + 1)  # fast grid size
        nyd = int(math.ceil((ymax - ymin) / stepsizex) + 1)  # slow grid size

        # Flatten as MATLAB reshape(..., [], 1) with Fortran order
        dset1line = dset1.T.reshape(-1, order='F')
        dset2line = pos_y.T.reshape(-1, order='F')  # fast indices
        dset3line = pos_x.T.reshape(-1, order='F')  # slow indices

        # Optional reference mask
        if args.use_ref and "PMT ref" in grp:
            thp = (2.0 - args.thres) if args.hotspots else args.thres
            refmask = refmask_find(np.asarray(grp["PMT ref"][()], dtype=np.float32),
                                   thres_percent=thp, spotlength=args.spotlength, njetspot0=50)
            maskline = refmask.T.reshape(-1, order='F')
            dset1line *= maskline

        # --- line selection by slow-axis parity (robust & simple) ---
        # Round slow-axis index to nearest integer line number (1-based)
        slow_idx = np.rint(dset3line).astype(np.int64)
        inb = (slow_idx >= 1) & (slow_idx <= nyd)
        if args.lines == "even":
            keep_line = (slow_idx % 2 == 0)
        elif args.lines == "odd":
            keep_line = (slow_idx % 2 == 1)
        else:
            keep_line = np.ones_like(slow_idx, dtype=bool)

        valid = (dset1line > 0.1) & inb & np.isfinite(dset2line) & np.isfinite(dset3line)
        sel = valid & keep_line
        if sel.sum() < 3:
            # Fallback to ALL lines if filtering is too aggressive
            sel = valid
        if sel.sum() < 3:
            raise RuntimeError("Too few valid points for interpolation.")

        v = dset1line[sel].astype(np.float32)
        x = dset3line[sel].astype(np.float32)  # first coord in griddata
        y = dset2line[sel].astype(np.float32)  # second coord

        # --- regular grid & linear scattered interpolation ---
        ss = np.tile(np.arange(1, nyd + 1, dtype=np.float32).reshape(-1, 1), (1, nxd))  # slow
        tt = np.tile(np.arange(1, nxd + 1, dtype=np.float32).reshape(1, -1), (nyd, 1))  # fast
        points = np.column_stack([x, y])

        imgbig = griddata(points, v, (ss, tt), method='linear')
        # Fill NaNs via nearest to reduce holes
        if np.isnan(imgbig).any():
            imgbig_nn = griddata(points, v, (ss, tt), method='nearest')
            m = np.isnan(imgbig)
            imgbig[m] = imgbig_nn[m]

        # Background from 25–75% percentile mean
        flat = imgbig.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            raise RuntimeError("Interpolation failed: no valid pixels.")
        q25, q75 = np.nanpercentile(flat, [25, 75])
        mid = flat[(flat >= q25) & (flat <= q75)]
        bgv = float(np.nanmean(mid)) if mid.size else float(np.nanmean(flat))

        # Center-crop/pad to (nx, ny)
        imgint = center_place((nx, ny), imgbig.astype(np.float32), bgv)

    # --- save PNG ---
    save_png_matlab_style(imgint, out_png, dpi=300, clip=(1, 99), colorbar=args.colorbar, show_axes=True)
    print(f"[PNG ] {out_png}")

    # --- save TIFF ---
    tifffile.imwrite(out_tiff,((imgint-np.min(imgint))/np.ptp(imgint)*65535).astype("uint16"))
    print(f"[TIFF ] {out_tiff}")

    # --- save HDF5 (compressed; layout aligned with previous scripts) ---
    dset1_out = np.flip(imgint.T, axis=1).astype(np.float32)  # Image1 = flip(imgint', 2)
    nx_out, ny_out = dset1_out.shape

    # metadata in µm (convert nm back to µm)
    XRange = (nx_out - 1) * (stepsizey / 1000.0)
    XStep  = (stepsizey / 1000.0)
    YRange = (ny_out - 1) * (stepsizex / 1000.0)
    YStep  = (stepsizex / 1000.0)

    with h5py.File(out_h5, "w") as g:
        grp_fs = g.create_group("FastAndSlowScanAxisPosition")
        grp_fs.create_dataset("FastScanAxisPosition",
                              data=np.full((nx_out * ny_out, 1), -400, dtype=np.float32),
                              compression="gzip", compression_opts=4)
        grp_fs.create_dataset("ScanSize",
                              data=np.array(nx_out * ny_out, dtype=np.float32))
        grp_fs.create_dataset("SlowScanAxisPosition",
                              data=np.full((nx_out * ny_out, 1),  400, dtype=np.float32),
                              compression="gzip", compression_opts=4)

        grp_im = g.create_group("ImageData1")
        grp_im.create_dataset("Image1", data=dset1_out,
                              compression="gzip", compression_opts=4)
        grp_im.create_dataset("Image2", data=np.ones_like(dset1_out, dtype=np.float32))
        grp_im.create_dataset("Image3", data=np.ones_like(dset1_out, dtype=np.float32))

        grp_info = g.create_group("ImageInfo")
        grp_info.create_dataset("Energy", data=np.array(energy_ev, dtype=np.float32))
        grp_info.create_dataset("ImageHeight", data=np.array(ny_out, dtype=np.float32))
        grp_info.create_dataset("ImageWidth", data=np.array(nx_out, dtype=np.float32))

        grp_scan = g.create_group("ScanningImageInfo")
        grp_scan.create_dataset("ClockSignalScaleFactor", data=np.array(80000.0, dtype=np.float32))
        grp_scan.create_dataset("DWellTime", data=np.array(1.0, dtype=np.float32))
        grp_scan.create_dataset("XRange", data=np.array(XRange, dtype=np.float32))
        grp_scan.create_dataset("XStep",  data=np.array(XStep,  dtype=np.float32))
        grp_scan.create_dataset("YRange", data=np.array(YRange, dtype=np.float32))
        grp_scan.create_dataset("YStep",  data=np.array(YStep,  dtype=np.float32))

    print(f"[HDF5] {out_h5} (Energy={energy_ev:.3f} eV)")


# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Optimized STXM regridder (PNG + HDF5 only)")
    ap.add_argument("--glob", default="*.h5", help="Input file pattern (default: *.h5)")
    ap.add_argument("--lines", choices=["even", "odd", "all"], default="even",
                    help="Keep only EVEN/ODD/ALL slow-axis lines (default: even)")
    ap.add_argument("--use-ref", dest="use_ref", action="store_true", default=True,
                    help="Use PMT reference to mask jet-noise/hot-spots (default True).")
    ap.add_argument("--no-ref", dest="use_ref", action="store_false",
                    help="Do not use PMT reference.")
    ap.add_argument("--hotspots", action="store_true", default=False,
                    help="Alternate scheme for PMT ref: thres <- 2 - thres.")
    ap.add_argument("--thres", type=float, default=0.96,
                    help="Threshold factor wrt background for ref mask (default 0.96).")
    ap.add_argument("--spotlength", type=int, default=30,
                    help="Jet-noise spot length in pixels (default 30).")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing *interp.hdf5 (default False).")
    ap.add_argument("--colorbar", action="store_true", default=False,
                    help="Add a slim colorbar on PNG.")
    ap.add_argument("--clip", nargs=2, type=float, metavar=("PLOW","PHIGH"), default=(1.0, 99.0),
                    help="Percentile clipping for PNG (default: 1 99).")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("No input .h5 files found.")
        return

    for p in files:
        try:
            process_file(p, args)
        except Exception as e:
            print(f"[Error] {os.path.basename(p)}: {e}")

if __name__ == "__main__":
    main()
