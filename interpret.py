#!/usr/bin/env python3
"""Interpretation layer: transcription -> drum PART.

transcribe.py recovers notes; this reconstructs the performance the way a
human transcriber would. Transcription noise is uncorrelated with musical
structure (random phantom/missing hits read as drunk); a real part repeats
its figures, crashes at phrase points, and holds a pulse. Three moves:

  1. crash discipline — keep accented crashes (local-max within ~1s), drop
     the wash re-triggers that produced a crash every 0.7 seconds;
  2. grid — snap notes to sixteenths in tempo-map beat space; the drummer's
     true microtiming is unrecoverable through this pipeline, so replace
     noisy timing with a clean grid (re-humanize downstream in drumgen);
  3. consensus — cluster bars by figure similarity; a bar close to its
     cluster's consensus snaps TO the consensus (restores the riff),
     an outlier bar stays untouched (fills survive).

Output: an interpreted .mid (tempo map preserved), and optionally the
per-bar figures as JSON for cell mining.

Usage:
    python interpret.py work/larsnet_all --name <track>.wav -o out.mid
    python interpret.py ... --dump-bars bars.json
    python interpret.py --selftest
"""
import argparse
import json

import numpy as np

from transcribe import (
    transcribe, write_tempo_map_midi,
    N_KICK, N_SNARE, N_CRASH, N_RIDE,
)

GRID = 4            # sixteenths per beat
BEATS_PER_BAR = 4
CRASH_MIN_GAP_S = 0.9
SNAP_SIM = 0.7      # bar joins its cluster consensus above this similarity
CLUSTER_SIM = 0.6   # bars cluster together above this similarity


# ── crash discipline ─────────────────────────────────────────────────────────

def restore_riding(events, bpm):
    """Crash-riding is timekeeping, not punctuation — a 0.9s gate on a washy
    section destroys the pulse (heard as 'child hitting pots' from bar ~85 of
    kidcrash_01). Cymbal-family onsets forming a regular run locked to a
    musical division (quarters or 8ths at track tempo, 4+ hits, IOI within
    ±25%) are flagged as riding, which exempts them from the punctuation
    gate. The crash/ride LABEL is left as the wash/discrete classifier set
    it — ear-verified on kidcrash_01 2:16: the drummer rides the CRASH
    there, so relabeling riding runs to ride cymbal was wrong."""
    fam = sorted((e for e in events if e["note"] in (N_CRASH, N_RIDE)),
                 key=lambda e: e["t"])
    ts = [e["t"] for e in fam]
    beat = 60.0 / bpm
    riding = set()
    for i in range(len(fam)):
        for div in (beat, beat / 2):
            j, run = i, [i]
            while j + 1 < len(fam) and abs((ts[j + 1] - ts[j]) - div) <= 0.25 * div:
                run.append(j + 1)
                j += 1
            if len(run) >= 4:
                riding.update(run)
    for k in riding:
        fam[k]["riding"] = True
    return len(riding)


def gate_crashes(events):
    """Greedy by velocity: a crash survives only if no stronger crash was
    already kept within CRASH_MIN_GAP_S. Kills wash re-triggers, keeps
    accents. Riding runs (flagged upstream) and other voices pass through."""
    crashes = [e for e in events
               if e["note"] == N_CRASH and not e.get("riding")]
    rest = [e for e in events
            if e["note"] != N_CRASH or e.get("riding")]
    kept = []
    for e in sorted(crashes, key=lambda e: -e["vel"]):
        if all(abs(e["t"] - k["t"]) >= CRASH_MIN_GAP_S for k in kept):
            kept.append(e)
    return sorted(rest + kept, key=lambda e: e["t"])


# ── beat-space grid ──────────────────────────────────────────────────────────

def to_beat_pos(t, beats):
    i = int(np.searchsorted(beats, t, side="right") - 1)
    if i < 0:
        span = beats[1] - beats[0]
        return (t - beats[0]) / span
    if i >= len(beats) - 1:
        span = beats[-1] - beats[-2]
        return (len(beats) - 1) + (t - beats[-1]) / span
    return i + (t - beats[i]) / (beats[i + 1] - beats[i])


def from_beat_pos(bp, beats):
    i = int(np.floor(bp))
    frac = bp - i
    if i < 0:
        return beats[0] + bp * (beats[1] - beats[0])
    if i >= len(beats) - 1:
        return beats[-1] + (bp - (len(beats) - 1)) * (beats[-1] - beats[-2])
    return beats[i] + frac * (beats[i + 1] - beats[i])


def quantize(events, beats):
    """Snap every note to the nearest sixteenth in beat space. Same-slot
    duplicates of one voice collapse to the loudest."""
    slots = {}
    for e in events:
        bp = to_beat_pos(e["t"], beats)
        q = round(bp * GRID) / GRID
        key = (q, e["note"])
        if key not in slots or e["vel"] > slots[key]["vel"]:
            e2 = dict(e)
            e2["bp"] = q
            slots[key] = e2
    out = sorted(slots.values(), key=lambda e: (e["bp"], e["note"]))
    for e in out:
        e["t"] = from_beat_pos(e["bp"], beats)
    return out


# ── consensus over bars ──────────────────────────────────────────────────────

def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def bar_figures(events):
    """{bar_index: {(slot_in_bar, note): velocity}} with slot in 0..15."""
    bars = {}
    for e in events:
        bar = int(e["bp"] // BEATS_PER_BAR)
        slot = int(round((e["bp"] - bar * BEATS_PER_BAR) * GRID))
        bars.setdefault(bar, {})[(slot, e["note"])] = e["vel"]
    return bars


def consensus_snap(events):
    """Cluster bars; snap near-consensus bars to consensus; leave outliers.

    Restores the repeating figure a listener latches onto ("direction")
    without erasing genuine fills/variations."""
    bars = bar_figures(events)
    idx = sorted(bars)
    keys = {b: frozenset(bars[b].keys()) for b in idx}

    unassigned = [b for b in idx if len(keys[b]) >= 3]
    clusters = []
    while unassigned:
        neigh = {b: [o for o in unassigned if jaccard(keys[b], keys[o]) >= CLUSTER_SIM]
                 for b in unassigned}
        seed = max(unassigned, key=lambda b: (len(neigh[b]), -b))
        members = neigh[seed]
        clusters.append(members)
        chosen = set(members)
        unassigned = [b for b in unassigned if b not in chosen]

    snapped, restored = 0, 0
    for members in clusters:
        if len(members) < 3:  # figure must recur to count as "the riff"
            continue
        counts = {}
        for b in members:
            for h in keys[b]:
                counts[h] = counts.get(h, 0) + 1
        need = max(2, (len(members) + 1) // 2)
        consensus = {h for h, c in counts.items() if c >= need}
        if not consensus:
            continue
        # median velocity per consensus hit across members that have it
        vel = {h: int(np.median([bars[b][h] for b in members if h in bars[b]]))
               for h in consensus}
        for b in members:
            if jaccard(keys[b], consensus) >= SNAP_SIM and keys[b] != consensus:
                bars[b] = {h: bars[b].get(h, vel[h]) for h in consensus}
                snapped += 1
                restored += len(consensus - keys[b])

    out = []
    for b, hits in bars.items():
        for (slot, note), v in hits.items():
            out.append({"bp": b * BEATS_PER_BAR + slot / GRID, "note": note, "vel": v})
    out.sort(key=lambda e: (e["bp"], e["note"]))
    return out, snapped, restored


# ── selftest ─────────────────────────────────────────────────────────────────

def selftest():
    # crash gate: 5 crashes inside 1s -> only the strongest survives
    ev = [{"t": 10.0 + i * 0.2, "note": N_CRASH, "vel": 60 + i} for i in range(5)]
    ev.append({"t": 10.4, "note": 36, "vel": 100})  # kick passes through
    g = gate_crashes(ev)
    crashes = [e for e in g if e["note"] == N_CRASH]
    assert len(crashes) == 1 and crashes[0]["vel"] == 64, \
        f"expected 1 strongest crash, got {[(c['t'], c['vel']) for c in crashes]}"
    assert any(e["note"] == 36 for e in g), "kick must pass through"

    # riding: 8 crashes at exact 8th spacing (169bpm) are flagged as riding,
    # keep their crash label, and ALL survive the gate; 3 scattered crashes
    # still gate down to the accent
    beat = 60.0 / 169
    ev2 = [{"t": 50.0 + i * beat / 2, "note": N_CRASH, "vel": 100} for i in range(8)]
    ev2 += [{"t": 80.0, "note": N_CRASH, "vel": 90},
            {"t": 80.3, "note": N_CRASH, "vel": 80},
            {"t": 80.5, "note": N_CRASH, "vel": 70}]
    n_r = restore_riding(ev2, 169)
    g2 = gate_crashes(ev2)
    ridden = [e for e in g2 if e.get("riding")]
    lone = [e for e in g2 if e["note"] == N_CRASH and not e.get("riding")]
    assert n_r == 8 and len(ridden) == 8, f"riding run must survive whole, got {len(ridden)}"
    assert all(e["note"] == N_CRASH for e in ridden), "riding keeps its crash label"
    assert len(lone) == 1 and lone[0]["vel"] == 90, "scattered crashes still gate"

    # quantize: two kicks 10 ms apart land in one sixteenth slot and collapse
    # to the louder; a snare 70 ms after the beat at 120 bpm rounds to the "e"
    beats = np.arange(16) * 0.5
    q = quantize([{"t": 2.0, "note": N_KICK, "vel": 80},
                  {"t": 2.01, "note": N_KICK, "vel": 100},
                  {"t": 2.07, "note": N_SNARE, "vel": 90}], beats)
    kicks = [e for e in q if e["note"] == N_KICK]
    assert len(kicks) == 1 and kicks[0]["vel"] == 100 and kicks[0]["bp"] == 4.0, \
        f"same-slot duplicates must collapse to the loudest, got {kicks}"
    snare = [e for e in q if e["note"] == N_SNARE][0]
    assert snare["bp"] == 4.25 and abs(snare["t"] - 2.125) < 1e-9, \
        f"snare must snap to the sixteenth after beat 4, got {snare}"

    # consensus: 4 near-identical bars, one missing a hit + one phantom.
    # Figure is realistically dense (8 hits): one error must not push the
    # bar below SNAP_SIM the way it would on a toy 4-hit figure.
    figure = [(0, 36), (2, 42), (4, 38), (6, 42),
              (8, 36), (10, 42), (12, 38), (14, 42)]
    events = []
    for bar in range(4):
        hits = list(figure)
        if bar == 2:
            hits = hits[:-1] + [(15, 49)]  # dropped one real hit, one phantom
        for slot, note in hits:
            events.append({"bp": bar * 4 + slot / 4, "note": note, "vel": 100})
    out, snapped, restored = consensus_snap(events)
    bars = bar_figures(out)
    figs = {b: frozenset(f.keys()) for b, f in bars.items()}
    assert figs[2] == figs[0], "noisy bar must snap to the consensus figure"
    assert snapped == 1 and restored >= 1
    print(f"OK — crash gate, riding, quantize, consensus snap ({snapped} bar "
          f"repaired, {restored} hit restored)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem_dir", nargs="?")
    ap.add_argument("--name")
    ap.add_argument("-o", "--out", default="part.mid")
    ap.add_argument("-s", "--sensitivity", type=float, default=0.5)
    ap.add_argument("--no-consensus", action="store_true",
                    help="skip bar-consensus repair (quantize + crash gate only)")
    ap.add_argument("--dump-bars", help="write per-bar figures JSON (cell mining)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.stem_dir and args.name):
        ap.error("need stem_dir and --name (or --selftest)")

    kept, beats, bpm = transcribe(args.stem_dir, args.name, args.sensitivity)
    n0 = len(kept)
    n_riding = restore_riding(kept, bpm)
    kept = gate_crashes(kept)
    n1 = len(kept)
    kept = quantize(kept, beats)
    n2 = len(kept)

    snapped = restored = 0
    if not args.no_consensus:
        kept, snapped, restored = consensus_snap(kept)
        for e in kept:
            e["t"] = from_beat_pos(e["bp"], beats)

    write_tempo_map_midi(kept, beats, bpm, args.out, tighten=False)

    if args.dump_bars:
        bars = bar_figures(kept)
        dump = {str(b): [[s, n, v] for (s, n), v in sorted(h.items())]
                for b, h in sorted(bars.items())}
        with open(args.dump_bars, "w") as f:
            json.dump({"bpm": bpm, "beats_per_bar": BEATS_PER_BAR,
                       "grid": GRID, "bars": dump}, f, indent=1)

    dur = kept[-1]["t"] - kept[0]["t"] if len(kept) > 1 else 1
    print(f"{n0} raw -> {n1} after crash gate ({n_riding} riding hits protected) "
          f"-> {n2} after quantize -> {len(kept)} final ({len(kept)/dur:.1f}/s) @ {bpm:.0f}bpm")
    print(f"  consensus: {snapped} bars snapped to their figure, "
          f"{restored} missing hits restored")
    print(f"  -> {args.out}" + (f" + {args.dump_bars}" if args.dump_bars else ""))


if __name__ == "__main__":
    main()
