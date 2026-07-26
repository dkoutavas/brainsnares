#!/usr/bin/env python3
"""One audition clip per track: all its mined cells laid out end to end.

Dragging 51 separate .mid files to judge them is the slow way. This builds
one clip per track — each cell played twice (you hear it as the loop it
would be), then a bar of silence, with a MIDI marker naming the cell and
its recurrence count so the DAW ruler tells you what you are listening to.

Cells are re-derived from work/bars/*.json with the same clustering
mine_cells.py uses, so sheet and cells always agree.

Usage:
    python contact_sheet.py work/bars/*.json -o work/sheets
    python contact_sheet.py --selftest
"""
import argparse
import glob
import os

import mido

from mine_cells import load_bars, cluster_figures

PPQ = 480
BEATS_PER_BAR = 4
PLAYS = 2          # times each cell repeats
GAP_BARS = 1       # silence after each cell
NOTE_TICKS = 48    # short, percussive


def build_sheet(figures, bpm, path, label):
    """figures: [(consensus {(slot,note):vel}, count)] -> one .mid"""
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=int(60_000_000 / bpm), time=0))
    tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tr.append(mido.MetaMessage("track_name", name=label, time=0))

    bar_ticks = BEATS_PER_BAR * PPQ
    events = []  # (tick, order, message)
    cursor = 0
    for rank, (fig, count) in enumerate(figures, 1):
        events.append((cursor, 0, mido.MetaMessage(
            "marker", text=f"{rank:02d} x{count}")))
        for rep in range(PLAYS):
            base = cursor + rep * bar_ticks
            for (slot, note), vel in sorted(fig.items()):
                on = base + int(slot * PPQ / 4)
                events.append((on, 1, mido.Message(
                    "note_on", channel=9, note=note, velocity=vel)))
                events.append((on + NOTE_TICKS, 1, mido.Message(
                    "note_off", channel=9, note=note, velocity=0)))
        cursor += (PLAYS + GAP_BARS) * bar_ticks

    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _order, msg in events:
        msg.time = tick - last
        last = tick
        tr.append(msg)
    mid.save(path)
    return cursor // bar_ticks


def selftest():
    import tempfile

    figs = [({(0, 36): 100, (4, 38): 90}, 7),
            ({(0, 36): 110, (8, 38): 95, (12, 42): 70}, 3)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.mid")
        bars = build_sheet(figs, 169, p, "test")
        mid = mido.MidiFile(p)
        ons = [m for m in mid.tracks[0] if m.type == "note_on"]
        marks = [m for m in mid.tracks[0] if m.type == "marker"]
        expect = PLAYS * sum(len(f) for f, _ in figs)
        assert len(ons) == expect, f"expected {expect} note_on, got {len(ons)}"
        assert len(marks) == len(figs), "one marker per cell"
        assert bars == len(figs) * (PLAYS + GAP_BARS), f"layout bars {bars}"
        # every note must land on a sixteenth of the grid
        t = 0
        for m in mid.tracks[0]:
            t += m.time
            if m.type == "note_on":
                assert t % (PPQ // 4) == 0, f"note off-grid at {t}"
    print(f"OK — {len(ons)} notes, {len(marks)} markers, {bars} bars, all on grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="*")
    ap.add_argument("-o", "--outdir", default="work/sheets")
    ap.add_argument("-n", "--top", type=int, default=6)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    files = [f for pat in args.jsons for f in glob.glob(pat)]
    if not files:
        ap.error("need bars.json files (or --selftest)")

    os.makedirs(args.outdir, exist_ok=True)
    for path in sorted(files):
        bars, bpm = load_bars(path)
        figs = cluster_figures(bars)[:args.top]
        stem = os.path.basename(path).replace(".json", "")
        if not figs:
            print(f"  {stem[:52]:52s} no recurring figures — skipped")
            continue
        out = os.path.join(args.outdir, f"{stem}__sheet.mid")
        n = build_sheet(figs, bpm, out, stem[:40])
        counts = ", ".join(f"x{c}" for _f, c in figs)
        print(f"  {stem[:52]:52s} {len(figs)} cells ({counts}) "
              f"{n} bars @ {bpm:.0f}bpm")
    print(f"\nsheets -> {args.outdir}/  "
          f"(each cell plays {PLAYS}x then {GAP_BARS} bar rest; markers name them)")


if __name__ == "__main__":
    main()
