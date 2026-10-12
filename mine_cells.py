#!/usr/bin/env python3
"""Mine drumgen cell candidates from interpreted bars.

Input: bars.json files from interpret.py (quantized, arbitrated, riding-
aware — the clean data the raw-transcription miner never had). Clusters
recurring bar figures per track and writes each cluster's consensus as a
single-bar .mid, named by recurrence count.

These are CANDIDATES: the author's ear curates, then imports keepers with
drumgen's own tool:

    cd ~/dev/drumgen
    python midi_reader.py ~/dev/brainsnares/work/cells/<cell>.mid --name <name> --auto-tag

Pure counting + consensus, deterministic. No inference.

Usage:
    python mine_cells.py work/bars/*.json -o work/cells
    python mine_cells.py --selftest
"""
import argparse
import glob
import json
import os

import numpy as np

from interpret import jaccard, CLUSTER_SIM

MIN_RECUR = 3   # figure must appear in >= this many bars to be a cell
MIN_HITS = 4    # ignore near-empty bars


def load_bars(path):
    d = json.load(open(path))
    bars = {}
    for b, hits in d["bars"].items():
        bars[int(b)] = {(s, n): v for s, n, v in hits}
    return bars, float(d["bpm"])


def cluster_figures(bars):
    """[(consensus {(slot,note):vel}, count)] sorted by recurrence."""
    idx = [b for b in sorted(bars) if len(bars[b]) >= MIN_HITS]
    keys = {b: frozenset(bars[b].keys()) for b in idx}
    unassigned = list(idx)
    out = []
    while unassigned:
        neigh = {b: [o for o in unassigned if jaccard(keys[b], keys[o]) >= CLUSTER_SIM]
                 for b in unassigned}
        seed = max(unassigned, key=lambda b: (len(neigh[b]), -b))
        members = neigh[seed]
        chosen = set(members)
        unassigned = [b for b in unassigned if b not in chosen]
        if len(members) < MIN_RECUR:
            continue
        counts = {}
        for b in members:
            for h in keys[b]:
                counts[h] = counts.get(h, 0) + 1
        need = max(2, (len(members) + 1) // 2)
        consensus = {}
        for h, c in counts.items():
            if c >= need:
                consensus[h] = int(np.median(
                    [bars[b][h] for b in members if h in bars[b]]))
        if len(consensus) >= MIN_HITS:
            out.append((consensus, len(members)))
    out.sort(key=lambda x: -x[1])
    return out


def write_cell(figure, bpm, path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    spb = 60.0 / bpm
    for (slot, note), vel in sorted(figure.items()):
        t = slot / 4 * spb
        inst.notes.append(pretty_midi.Note(
            velocity=vel, pitch=note, start=t, end=t + 0.05))
    pm.instruments.append(inst)
    pm.write(path)


def selftest():
    bars = {}
    fig = {(0, 36): 110, (4, 38): 100, (8, 36): 105, (12, 38): 100, (2, 42): 80}
    for b in range(5):
        bars[b] = dict(fig)
    bars[5] = {(0, 49): 90, (1, 41): 70, (2, 41): 70, (3, 47): 80}  # one-off fill
    got = cluster_figures(bars)
    assert len(got) == 1, f"only the recurring figure qualifies, got {len(got)}"
    consensus, count = got[0]
    assert count == 5 and set(consensus) == set(fig), "consensus must match the riff"
    print(f"OK — 1 figure x{count}, fill excluded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="*", help="bars.json files")
    ap.add_argument("-o", "--outdir", default="work/cells")
    ap.add_argument("-n", "--top", type=int, default=6, help="max cells per track")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    files = [f for pat in args.jsons for f in glob.glob(pat)]
    if not files:
        ap.error("need bars.json files (or --selftest)")

    os.makedirs(args.outdir, exist_ok=True)
    total = 0
    for path in sorted(files):
        bars, bpm = load_bars(path)
        # short slug: first two words + last word of the track name
        stem = os.path.basename(path).replace(".json", "")
        words = stem.split("_")
        slug = "_".join(words[:2] + words[-1:]) if len(words) > 3 else stem
        figs = cluster_figures(bars)[:args.top]
        for rank, (fig, count) in enumerate(figs, 1):
            name = f"{slug}_{rank:02d}_x{count}_{int(round(bpm))}bpm.mid"
            write_cell(fig, bpm, os.path.join(args.outdir, name))
        total += len(figs)
        print(f"  {stem[:56]:56s} {len(figs)} cells "
              f"(top recurs {figs[0][1]}x)" if figs else f"  {stem[:56]:56s} none recur")
    print(f"\n{total} cell candidates -> {args.outdir}/")
    # per-file with an explicit --name: bulk directory import takes no --name,
    # yielding doubled "_<bpm>bpm_<bpm>bpm" names and no provenance tags
    print("import keepers (per file):  cd ~/dev/drumgen && python midi_reader.py "
          f"{os.path.abspath(args.outdir)}/<cell>.mid --name <name> --auto-tag")


if __name__ == "__main__":
    main()
