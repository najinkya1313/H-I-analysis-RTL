#!/usr/bin/env python3
"""
rtl_hi.py -- run rtl_power, then immediately plot the spectrum.

Usage
-----
  # acquire + plot. You MUST say whether the bias tee (SAWbird power) is on or off:
  python rtl_hi.py obs1.csv --bias-tee on
  python rtl_hi.py obs1.csv --bias-tee off

  # override rtl_power settings
  python rtl_hi.py obs1.csv --bias-tee on -f 1419.5M:1421.5M:4k -g 50 -i 120s

  # skip acquisition, just plot an existing file (no bias-tee choice needed)
  python rtl_hi.py obs1.csv --plot-only

  # or keep your own rtl_power command and chain it in the shell
  rtl_power -f 1419.5M:1421.5M:4k -g 50 -i 60s -T -1 obs1.csv && python rtl_hi.py obs1.csv --plot-only

Every plot is also saved as a PNG next to the CSV (obs1_raw.png,
obs1_calibrated.png), so it works fine without a GUI (use --no-show to
suppress windows explicitly).

Window 1: raw spectrum (always).
Window 2: calibrated temperature (only if ground.csv and sky.csv exist and
          the file being plotted is not itself ground.csv / sky.csv).
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Constants -- edit these
# --------------------------------------------------------------------------
F0_MHZ = 1420.405751768          # HI rest frequency
C_KMS = 299792.458               # speed of light, km/s
T_SKY = 5.0                      # K, cold-sky reference
T_GROUND = 300.0                 # K, ground reference
HI_MASK_MHZ = (1420.000, 1420.700)   # HI contamination in sky.csv, interpolated over
                                     # (override on the command line with --mask LO HI)
FIGSIZE = (9, 5.5)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_rtl_power_csv(path):
    """Read an rtl_power CSV (no header row).

    Columns: date, time, hz_low, hz_high, hz_step, samples, dB, dB, ...
    Frequencies are built from hz_low + hz_step * i, so no hard-coded bin size.
    Repeated bins (multiple rows / sweeps at the same frequency) are averaged
    in linear power. Returns (freq_hz, power_dB), sorted by frequency.
    """
    freqs, powers = [], []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 7:
                continue
            row = [x.strip() for x in row]
            f_lo, step = float(row[2]), float(row[4])
            p = np.array([float(x) for x in row[6:] if x], dtype=float)
            freqs.append(f_lo + step * np.arange(p.size))
            powers.append(p)
    if not freqs:
        raise ValueError(f"No data rows found in {path}")

    f = np.concatenate(freqs)
    p = np.concatenate(powers)
    ok = np.isfinite(f) & np.isfinite(p)
    f, p = f[ok], p[ok]

    f_u, inv = np.unique(np.round(f), return_inverse=True)
    inv = inv.ravel()
    p_lin = np.bincount(inv, weights=10 ** (p / 10)) / np.bincount(inv)
    return f_u, 10 * np.log10(p_lin)


# --------------------------------------------------------------------------
# Calibration (same maths as your plot_velocity)
# --------------------------------------------------------------------------
def calibrate(freq, p_src_db, ground_path, sky_path, mask_hz):
    """Return (T_source [K], mask) on the source frequency grid.

    ground/sky are resampled onto the source grid if their grids differ.
    """
    fg, pg = read_rtl_power_csv(ground_path)
    fs, ps = read_rtl_power_csv(sky_path)

    for name, fc in (("ground", fg), ("sky", fs)):
        if fc.min() > freq.min() + 1 or fc.max() < freq.max() - 1:
            print(f"WARNING: {name}.csv does not cover the full frequency range "
                  f"of the data; edges will be held flat.")

    P_ground = np.interp(freq, fg, 10 ** (pg / 10))
    P_sky = np.interp(freq, fs, 10 ** (ps / 10))
    P_src = 10 ** (p_src_db / 10)

    mask = (freq >= mask_hz[0]) & (freq <= mask_hz[1])

    with np.errstate(divide="ignore", invalid="ignore"):
        Y = P_ground / P_sky
        T_r = (T_SKY * Y - T_GROUND) / (1 - Y)

        # straight-line interpolation of T_r across the HI-contaminated region
        T_r_clean = T_r.copy()
        if mask.any() and (~mask).any():
            T_r_clean[mask] = np.interp(freq[mask], freq[~mask], T_r[~mask])

        T_src = (T_GROUND + T_r_clean) / (P_ground / P_src) - T_r_clean

    return T_src, mask


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _f2v(f_mhz):
    return C_KMS * (F0_MHZ - f_mhz) / F0_MHZ


def _v2f(v_kms):
    return F0_MHZ * (1 - v_kms / C_KMS)


def _finish_axes(fig, ax, title):
    """Frequency axis on the bottom, velocity axis offset below it."""
    ax.set_xlabel("Frequency (MHz)")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(alpha=0.3)
    ax.set_title(title)

    # Secondary axis placed at -0.2 in axes coordinates: well clear of the
    # primary tick labels *and* the primary x label.
    sec = ax.secondary_xaxis(-0.2, functions=(_f2v, _v2f))
    sec.set_xlabel("Velocity (km/s)")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.27)


def plot_raw(freq_hz, p_db, title):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(freq_hz / 1e6, p_db, "o", ms=2)
    ax.axvline(F0_MHZ, color="gray", ls=":", lw=1)
    ax.set_ylabel("Power (dB, uncalibrated)")
    _finish_axes(fig, ax, title)
    fig.canvas.manager.set_window_title("Raw spectrum")
    return fig


def plot_calibrated(freq_hz, T_src, mask, title):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(freq_hz / 1e6, T_src, lw=1)
    ax.axvline(F0_MHZ, color="gray", ls=":", lw=1)
    ax.set_ylabel("Temperature (K)")
    _finish_axes(fig, ax, title)
    fig.canvas.manager.set_window_title("Calibrated spectrum")
    return fig


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run rtl_power and plot the result.")
    ap.add_argument("filename", help="output CSV (written by rtl_power, or existing with --plot-only)")
    ap.add_argument("-f", "--freq", default="1419.5M:1421.5M:4k", help="rtl_power -f argument")
    ap.add_argument("-g", "--gain", default="50", help="rtl_power -g argument")
    ap.add_argument("-i", "--interval", default="60s", help="rtl_power -i argument")
    ap.add_argument("-T", "--bias-tee", choices=["on", "off"], default=None,
                    help="REQUIRED when acquiring: 'on' adds -T to rtl_power (powers the SAWbird "
                         "through the coax), 'off' leaves it out. Not needed with --plot-only.")
    ap.add_argument("--plot-only", action="store_true", help="skip rtl_power, just plot the file")
    ap.add_argument("--mask", nargs=2, type=float, metavar=("LO_MHZ", "HI_MHZ"),
                    default=HI_MASK_MHZ, help="HI-contaminated range in sky.csv (MHz)")
    ap.add_argument("--cal-dir", default=None,
                    help="directory holding ground.csv / sky.csv (default: directory of the data file)")
    ap.add_argument("--no-show", action="store_true",
                    help="don't open plot windows (PNGs are still saved)")
    ap.add_argument("--dpi", type=int, default=150, help="PNG resolution (default 150)")
    args = ap.parse_args()

    # ---- acquisition ----
    if not args.plot_only:
        # Bias tee is deliberately mandatory: no default, so you decide every run.
        if args.bias_tee is None:
            ap.error("--bias-tee {on,off} is required when acquiring "
                     "(on = power the SAWbird via the SDR, off = no bias voltage)")
        if shutil.which("rtl_power") is None:
            sys.exit("rtl_power not found in PATH (install rtl-sdr, or use --plot-only).")
        cmd = ["rtl_power", "-f", args.freq, "-g", args.gain, "-i", args.interval]
        if args.bias_tee == "on":
            cmd.append("-T")          # rtl_power: enable bias-T (GPIO pin 0, RTL-SDR Blog dongles)
        cmd += ["-1", args.filename]
        print(f"Bias tee: {args.bias_tee.upper()}")
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"rtl_power failed (exit code {e.returncode}).")
        except KeyboardInterrupt:
            sys.exit("\nAcquisition interrupted.")

    if not os.path.isfile(args.filename):
        sys.exit(f"{args.filename} not found.")

    # ---- raw plot ----
    freq, p_db = read_rtl_power_csv(args.filename)
    name = os.path.basename(args.filename)
    stem = os.path.splitext(os.path.abspath(args.filename))[0]   # saved PNGs sit next to the CSV
    fig_raw = plot_raw(freq, p_db, f"{name}  ({freq.min()/1e6:.3f}-{freq.max()/1e6:.3f} MHz, {freq.size} bins)")
    fig_raw.savefig(stem + "_raw.png", dpi=args.dpi)
    print(f"Saved {stem}_raw.png")

    # ---- calibrated plot ----
    cal_dir = args.cal_dir or os.path.dirname(os.path.abspath(args.filename))
    ground = os.path.join(cal_dir, "ground.csv")
    sky = os.path.join(cal_dir, "sky.csv")
    is_cal_file = name.lower() in ("ground.csv", "sky.csv")

    if is_cal_file:
        print("Calibration file -- skipping temperature calibration.")
    elif os.path.isfile(ground) and os.path.isfile(sky):
        mask_hz = (args.mask[0] * 1e6, args.mask[1] * 1e6)
        T_src, mask = calibrate(freq, p_db, ground, sky, mask_hz)
        fig_cal = plot_calibrated(freq, T_src, mask, f"{name} -- calibrated")
        fig_cal.savefig(stem + "_calibrated.png", dpi=args.dpi)
        print(f"Saved {stem}_calibrated.png")
    else:
        missing = [n for n, p in (("ground.csv", ground), ("sky.csv", sky)) if not os.path.isfile(p)]
        print(f"No calibration: missing {', '.join(missing)} in {cal_dir}")

    # Only open windows if requested AND the backend can actually display them
    # (headless Linux falls back to a non-interactive backend like Agg).
    non_gui = {"agg", "pdf", "svg", "ps", "cairo", "pgf", "template"}
    if args.no_show or plt.get_backend().lower() in non_gui:
        print("Not opening windows (headless backend or --no-show).")
    else:
        plt.show()


if __name__ == "__main__":
    main()
