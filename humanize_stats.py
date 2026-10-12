#!/usr/bin/env python3
"""Measure human feel from real recordings, for drumgen's humanizer constants.

SUPERSEDED. On this material the beat grid is unreliable (alignment 0.78
where 1.0 is random), so the timing statistics are contaminated. Do not derive
drumgen's TIMING_TENDENCIES from this output. Velocity ordering survived but
mix compression confounds it; treat it as a hint. See README, "What failed".

Reads LarsNet per-drum stems, detects onsets on each, snaps them to a beat
grid, and reports how far off the grid real drummers actually play and how
much their velocities vary. Those are exactly the numbers hand-tuned in
drumgen's humanizer.py (TIMING_TENDENCIES, INSTRUMENT_VARIANCE, the 6ms
jitter sigma, CONTOUR_OFFSETS).

Why this survives bad transcription: these are DISTRIBUTIONS over thousands
of hits. Individual misdetections shift a mean by a fraction of a ms, whereas
they destroy note-level pattern matching outright. Measurement, not
inference — no model, no generation.

    python humanize_stats.py work/larsnet_all -o stats.json
    python humanize_stats.py --selftest
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# larsnet stem folder -> (drumgen instrument name, min gap ms)
STEMS = {
    "kick":    ("kick", 55),
    "snare":   ("snare", 45),
    "toms":    ("tom_mid", 60),
    "hihat":   ("hihat_closed", 30),
    "cymbals": ("crash_1", 55),
}
HOP = 256
GRID = 4  # sixteenths


def stem_onsets(path, min_gap_ms, sensitivity=0.5):
    """(times, strengths) for one isolated drum stem."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    if not np.any(y):
        return np.array([]), np.array([]), None
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    peak = env.max()
    if peak <= 0:
        return np.array([]), np.array([]), None
    env = env / peak
    frames = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=12, post_avg=12,
        delta=max(0.005, 0.10 * (1.0 - sensitivity)),
        wait=max(1, int(round(min_gap_ms / 1000 * sr / HOP))),
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    return times, env[frames], (y, sr)


def grid_offsets(times, beats):
    """Signed ms each hit lands from its nearest sixteenth, plus its sub index.

    Positive = late (behind the beat), negative = early (pushing).

    Also returns the indices of `times` that fell inside the beat grid —
    hits before the first beat or after the last are dropped, so callers
    must use these to line up any parallel array (e.g. velocities).
    """
    offs, subs, keep = [], [], []
    if len(beats) < 2:
        return np.array([]), np.array([]), np.array([], dtype=int)
    for idx, t in enumerate(times):
        i = int(np.searchsorted(beats, t, side="right") - 1)
        if i < 0 or i >= len(beats) - 1:
            continue
        span = beats[i + 1] - beats[i]
        if span <= 0:
            continue
        frac = (t - beats[i]) / span
        nearest = round(frac * GRID) / GRID
        # offset in ms relative to that sixteenth
        offs.append((frac - nearest) * span * 1000.0)
        subs.append(nearest % 1.0)
        keep.append(idx)
    return np.array(offs), np.array(subs), np.array(keep, dtype=int)


def analyse_track(stem_dir, name):
    """Per-instrument timing/velocity stats for one track."""
    import librosa

    # beat grid from the snare stem where possible (clearest pulse)
    ref = None
    for pref in ("snare", "kick", "hihat"):
        p = os.path.join(stem_dir, pref, name)
        if os.path.exists(p):
            ref = p
            break
    if ref is None:
        return None
    y, sr = librosa.load(ref, sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) < 8:
        return None

    out = {"tempo": tempo, "instruments": {}}
    for stem, (inst, gap) in STEMS.items():
        p = os.path.join(stem_dir, stem, name)
        if not os.path.exists(p):
            continue
        times, strengths, _ = stem_onsets(p, gap)
        if len(times) < 20:  # too few hits to say anything
            continue
        offs, subs, keep = grid_offsets(times, beats)
        if len(offs) < 20:
            continue
        # velocity proxy: onset strength scaled to MIDI range, restricted to
        # the hits that actually landed inside the beat grid
        vels = np.clip(45 + strengths[keep] * 170, 30, 127)
        by_sub = {}
        for s in (0.0, 0.25, 0.5, 0.75):
            m = np.isclose(subs, s)
            if m.sum() >= 5:
                by_sub[str(s)] = float(vels[m].mean())
        out["instruments"][inst] = {
            "hits": int(len(offs)),
            "timing_mean_ms": float(offs.mean()),
            "timing_sigma_ms": float(offs.std()),
            "vel_mean": float(vels.mean()),
            "vel_sigma": float(vels.std()),
            "vel_by_sub": by_sub,
        }
    return out


def aggregate(tracks):
    """Pool per-track stats into one set of constants, weighted by hit count."""
    acc = defaultdict(lambda: defaultdict(list))
    for tr in tracks:
        for inst, s in tr["instruments"].items():
            w = s["hits"]
            acc[inst]["w"].append(w)
            for k in ("timing_mean_ms", "timing_sigma_ms", "vel_mean", "vel_sigma"):
                acc[inst][k].append(s[k])
            for sub, v in s["vel_by_sub"].items():
                acc[inst][f"sub_{sub}"].append(v)

    out = {}
    for inst, d in acc.items():
        w = np.array(d["w"], dtype=float)
        row = {"hits": int(w.sum()), "tracks": len(w)}
        for k in ("timing_mean_ms", "timing_sigma_ms", "vel_mean", "vel_sigma"):
            row[k] = float(np.average(np.array(d[k]), weights=w))
        subs = {}
        for sub in ("0.0", "0.25", "0.5", "0.75"):
            vals = d.get(f"sub_{sub}")
            if vals:
                subs[sub] = float(np.mean(vals))
        if subs:
            base = np.mean(list(subs.values()))
            row["vel_contour_offset"] = {k: round(v - base, 1) for k, v in subs.items()}
        out[inst] = row
    return out


def selftest():
    """Hits deliberately placed 10ms late must measure as ~+10ms, and the
    sign convention must be right (late = positive)."""
    tempo = 120.0
    spb = 60.0 / tempo
    beats = np.arange(0, 9) * spb
    late_ms = 10.0
    times = np.array([b + late_ms / 1000.0 for b in beats[:8]])
    offs, subs, keep = grid_offsets(times, beats)
    assert len(keep) == len(offs), "keep indices must align with offsets"
    assert len(offs) >= 6, f"expected offsets, got {len(offs)}"
    assert abs(offs.mean() - late_ms) < 1.5, \
        f"10ms-late hits should measure ~+10ms, got {offs.mean():.1f}"
    assert np.allclose(subs[:6], 0.0), "on-beat hits should map to sub 0.0"

    early = np.array([b + spb / 4 - 0.008 for b in beats[:8]])
    offs2, subs2, _ = grid_offsets(early, beats)
    assert offs2.mean() < 0, "early hits must measure negative"
    assert np.allclose(subs2[:6], 0.25), "hits near the e should map to sub 0.25"
    print(f"OK — late={offs.mean():+.1f}ms, early={offs2.mean():+.1f}ms, subs correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem_dir", nargs="?", help="larsnet output root")
    ap.add_argument("-o", "--out", default="humanize_stats.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.stem_dir:
        ap.error("need stem_dir (or --selftest)")

    names = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(args.stem_dir, "kick", "*.wav")))
    if not names:
        print(f"no stems found under {args.stem_dir}/kick/")
        return

    tracks = []
    for n in names:
        r = analyse_track(args.stem_dir, n)
        if r:
            r["track"] = n
            tracks.append(r)
            print(f"  {n[:52]:52s} ~{r['tempo']:5.1f}bpm  "
                  f"{len(r['instruments'])} instruments")

    if not tracks:
        print("nothing analysable")
        return

    agg = aggregate(tracks)
    with open(args.out, "w") as f:
        json.dump({"per_track": tracks, "aggregate": agg}, f, indent=2)

    print(f"\n=== measured across {len(tracks)} tracks -> {args.out} ===")
    print(f"{'instrument':14s} {'hits':>6s} {'timing':>9s} {'jitter':>8s} {'vel sigma':>10s}")
    for inst, r in sorted(agg.items(), key=lambda x: -x[1]["hits"]):
        print(f"{inst:14s} {r['hits']:6d} {r['timing_mean_ms']:+8.1f}ms "
              f"{r['timing_sigma_ms']:7.1f}ms {r['vel_sigma']:9.1f}")


if __name__ == "__main__":
    main()
