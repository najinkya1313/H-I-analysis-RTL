#!/usr/bin/env python3
"""
rtl_live.py -- watch the spectrum build up while rtl_power integrates.

How it works
------------
rtl_power only writes a row when an integration interval finishes, so instead
of one 60 s integration we ask it for short intervals (default 1 s), stream the
rows from its stdout, and keep a running average in *linear power*. Averaging
N one-second spectra is equivalent (to within a little retune dead time) to one
N-second integration, and you get to watch the noise beat down in real time.

When it finishes (or you hit Ctrl-C / close the window) the averaged spectrum is
written to FILENAME in the normal rtl_power CSV format, so ground.csv / sky.csv
recorded this way work with rtl_hi.py and with the calibration here.

Needs rtl_hi.py in the same directory (re-uses its calibration + plot styling).

Usage
-----
  python rtl_live.py obs1.csv                 # 60 s total, 1 s updates
  python rtl_live.py obs1.csv -d 5m           # 5 minutes
  python rtl_live.py ground.csv -d 2m         # record a calibration file live
  python rtl_live.py obs1.csv -d 10m -i 2s    # slower, smoother updates

Window 1: live raw spectrum (grey = latest single interval, dots = running mean)
Window 2: live calibrated temperature (only if ground.csv + sky.csv exist and
          the file you're recording isn't one of them)
"""
import argparse
import contextlib
import datetime
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import matplotlib.pyplot as plt

from rtl_hi import F0_MHZ, FIGSIZE, HI_MASK_MHZ, calibrate, _finish_axes


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def parse_time(s):
    """'90', '90s', '5m', '1.5h' -> seconds (same suffixes rtl_power accepts)."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([smh]?)\s*", s)
    if not m:
        raise ValueError(f"can't parse time '{s}' (use e.g. 60s, 5m, 1h)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


class Integrator:
    """Running linear-power average of streamed rtl_power rows (per hop)."""

    def __init__(self):
        self.hops = {}   # hz_low -> dict(hi, step, samples, sum, n, last)

    def add(self, line):
        row = [x.strip() for x in line.strip().split(",")]
        if len(row) < 7:
            return False
        try:
            lo, hi, step, ns = (float(row[i]) for i in range(2, 6))
            p = np.array([float(x) for x in row[6:] if x])
        except ValueError:
            return False
        with np.errstate(over="ignore"):
            lin = 10 ** (p / 10)
        h = self.hops.get(lo)
        if h is None:
            self.hops[lo] = dict(hi=hi, step=step, samples=ns, sum=lin, n=1, last=p)
        elif lin.size == h["sum"].size:
            h["sum"] += lin
            h["n"] += 1
            h["samples"] += ns
            h["last"] = p
        else:
            return False
        return True

    @property
    def sweeps(self):
        return min((h["n"] for h in self.hops.values()), default=0)

    def spectra(self):
        """Return (freq_hz, mean_dB, latest_dB), hops concatenated in order."""
        f, avg, last = [], [], []
        for lo in sorted(self.hops):
            h = self.hops[lo]
            f.append(lo + h["step"] * np.arange(h["sum"].size))
            with np.errstate(divide="ignore"):
                avg.append(10 * np.log10(h["sum"] / h["n"]))
            last.append(h["last"])
        return np.concatenate(f), np.concatenate(avg), np.concatenate(last)

    def write_csv(self, path):
        """Write the averaged spectrum in rtl_power CSV format (one row per hop)."""
        now = datetime.datetime.now()
        with open(path, "w") as fh:
            for lo in sorted(self.hops):
                h = self.hops[lo]
                with np.errstate(divide="ignore"):
                    p = 10 * np.log10(h["sum"] / h["n"])
                fh.write(f"{now:%Y-%m-%d}, {now:%H:%M:%S}, {lo:.0f}, {h['hi']:.0f}, "
                         f"{h['step']:.2f}, {h['samples']:.0f}, "
                         + ", ".join(f"{x:.2f}" for x in p) + "\n")


def _set_ylim(ax, y, lo_pct=0, hi_pct=100, pad=0.15, min_pad=0.5):
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    lo, hi = np.percentile(y, [lo_pct, hi_pct])
    m = max(min_pad, pad * (hi - lo))
    ax.set_ylim(lo - m, hi + m)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Live-integrating spectrum display for rtl_power.")
    ap.add_argument("filename", help="output CSV (averaged spectrum, rtl_power format)")
    ap.add_argument("-f", "--freq", default="1419.5M:1421.5M:4k", help="rtl_power -f argument")
    ap.add_argument("-g", "--gain", default="50", help="rtl_power -g argument")
    ap.add_argument("-i", "--interval", default="1s", help="per-update interval (default 1s)")
    ap.add_argument("-d", "--duration", default="60s", help="total integration time, e.g. 60s, 5m")
    ap.add_argument("--mask", nargs=2, type=float, metavar=("LO_MHZ", "HI_MHZ"),
                    default=HI_MASK_MHZ, help="HI-contaminated range in sky.csv (MHz)")
    ap.add_argument("--cal-dir", default=None,
                    help="directory holding ground.csv / sky.csv (default: directory of the output file)")
    ap.add_argument("--no-show", action="store_true", help="don't keep windows open at the end")
    ap.add_argument("--dpi", type=int, default=150, help="PNG resolution (default 150)")
    args = ap.parse_args()

    if shutil.which("rtl_power") is None:
        sys.exit("rtl_power not found in PATH (install rtl-sdr).")
    try:
        interval_s = parse_time(args.interval)
        total_s = parse_time(args.duration)
    except ValueError as e:
        sys.exit(str(e))

    name = os.path.basename(args.filename)
    stem = os.path.splitext(os.path.abspath(args.filename))[0]
    cal_dir = args.cal_dir or os.path.dirname(os.path.abspath(args.filename))
    ground = os.path.join(cal_dir, "ground.csv")
    sky = os.path.join(cal_dir, "sky.csv")
    do_cal = (name.lower() not in ("ground.csv", "sky.csv")
              and os.path.isfile(ground) and os.path.isfile(sky))
    mask_hz = (args.mask[0] * 1e6, args.mask[1] * 1e6)
    print("Live calibration: " + ("ON" if do_cal else "off (need ground.csv + sky.csv, and not recording one of them)"))

    # rtl_power writes to stdout when the filename is '-'. stdio fully buffers when piped,
    # which would make the display lurch; stdbuf -oL forces line buffering.
    cmd = ["rtl_power", "-f", args.freq, "-g", args.gain, "-i", args.interval,
           "-e", args.duration, "-"]
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL"] + cmd
    else:
        print("WARNING: stdbuf not found; updates may arrive in bursts.")
    print("Running:", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    q = queue.Queue()

    def reader():
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    integ = Integrator()
    fig_raw = fig_cal = None
    ln_last = ln_avg = ln_cal = None
    first_cal_call = True

    def title(prefix):
        done = min(integ.sweeps * interval_s, total_s)
        return f"{name} - {prefix}: {done:.0f} / {total_s:.0f} s ({integ.sweeps} spectra)"

    def refresh(prefix="live", show=True):
        nonlocal fig_raw, fig_cal, ln_last, ln_avg, ln_cal, first_cal_call
        freq, avg, last = integ.spectra()
        x = freq / 1e6

        if fig_raw is None:   # build windows on first data
            fig_raw, ax = plt.subplots(figsize=FIGSIZE)
            ln_last, = ax.plot(x, last, "-", lw=0.6, color="0.7", label="latest interval")
            ln_avg, = ax.plot(x, avg, "o", ms=2, label="running mean")
            ax.axvline(F0_MHZ, color="gray", ls=":", lw=1)
            ax.set_ylabel("Power (dB, uncalibrated)")
            ax.legend(loc="upper right")
            _finish_axes(fig_raw, ax, title(prefix))
            fig_raw.canvas.manager.set_window_title("Live raw spectrum")
            if show:
                fig_raw.show()

        ln_last.set_data(x, last)
        ln_avg.set_data(x, avg)
        ax = fig_raw.axes[0]
        ax.set_xlim(x.min(), x.max())
        _set_ylim(ax, avg)
        ax.set_title(title(prefix))

        if do_cal:
            # calibrate() prints coverage warnings; show them once, not every update
            ctx = contextlib.nullcontext() if first_cal_call else contextlib.redirect_stdout(io.StringIO())
            with ctx:
                T, mask = calibrate(freq, avg, ground, sky, mask_hz)
            first_cal_call = False
            if fig_cal is None:
                fig_cal, axc = plt.subplots(figsize=FIGSIZE)
                ln_cal, = axc.plot(x, T, lw=1)
                if mask.any():
                    axc.axvspan(x[mask].min(), x[mask].max(), color="orange", alpha=0.15,
                                label="interpolated in sky cal")
                    axc.legend(loc="upper right")
                axc.axvline(F0_MHZ, color="gray", ls=":", lw=1)
                axc.set_ylabel("Temperature (K)")
                _finish_axes(fig_cal, axc, title(prefix))
                fig_cal.canvas.manager.set_window_title("Live calibrated spectrum")
                if show:
                    fig_cal.show()
            ln_cal.set_data(x, T)
            axc = fig_cal.axes[0]
            axc.set_xlim(x.min(), x.max())
            _set_ylim(axc, T, lo_pct=2, hi_pct=98, pad=0.3, min_pad=1.0)
            axc.set_title(title(prefix))

    stopped_early = False
    try:
        finished = False
        while not finished:
            got = False
            while True:
                try:
                    line = q.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    finished = True
                    break
                got |= integ.add(line)
            if got:
                refresh()
            if fig_raw is not None and not plt.fignum_exists(fig_raw.number):
                stopped_early = True      # user closed the live window
                break
            plt.pause(0.05)
    except KeyboardInterrupt:
        stopped_early = True
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    if integ.sweeps == 0:
        sys.exit("No spectra received from rtl_power (is the dongle plugged in / free?).")

    # ---- save results ----
    integ.write_csv(args.filename)
    print(f"{'Stopped early; saved' if stopped_early else 'Saved'} {integ.sweeps} averaged spectra -> {args.filename}")

    window_open = fig_raw is not None and plt.fignum_exists(fig_raw.number)
    if not window_open:      # recreate figures if the user closed them, just to save PNGs
        fig_raw = fig_cal = None
    refresh("final", show=window_open)
    fig_raw.savefig(stem + "_raw.png", dpi=args.dpi)
    print(f"Saved {stem}_raw.png")
    if fig_cal is not None:
        fig_cal.savefig(stem + "_calibrated.png", dpi=args.dpi)
        print(f"Saved {stem}_calibrated.png")

    non_gui = {"agg", "pdf", "svg", "ps", "cairo", "pgf", "template"}
    if window_open and not args.no_show and plt.get_backend().lower() not in non_gui:
        plt.show()


if __name__ == "__main__":
    main()
