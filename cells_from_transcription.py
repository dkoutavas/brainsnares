#!/usr/bin/env python3
"""Mine repeated 1-bar drum figures out of a transcription.

Input:  a transcribed .mid (absolute times, messy) + the drums stem it came
        from (used for beat tracking, which is far more reliable on audio).
Output: the most-REPEATED quantized bar patterns, each written as its own
        .mid, ready for drumgen's `midi_reader.py --auto-tag`.

Why rank by repetition: a real drum part restates its figures across a song;
transcription noise does not. Counting recurrence is therefore both the
vocabulary extractor and the noise filter. Pure counting — no inference.

Deterministic: same input always yields the same cells.

Usage:
    python cells_from_transcription.py drums.wav transcript.mid -o cells/
    python cells_from_transcription.py --selftest
"""
import argparse
import os
from collections import Counter, defaultdict

import numpy as np

GRID = 4  # subdivisions per beat (4 = sixteenth notes)
NAMES = {36: "kick", 38: "snare", 42: "hat", 49: "crash"}


def beat_grid(wav_path):
    """Beat positions (seconds) + tempo, tracked from the audio."""
    import librosa

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", trim=False)
    return np.asarray(beats, dtype=float), float(np.atleast_1d(tempo)[0])


def quantize_to_bars(hits, beats, beats_per_bar=4):
    """Snap hits onto a bar/sub grid.

    Returns {bar_index: frozenset((beat, sub, note))} where beat is 1-indexed
    and sub is the sixteenth offset (0.0, 0.25, 0.5, 0.75).
    """
    bars = defaultdict(set)
    if len(beats) < beats_per_bar + 1:
        return bars

    for t, note, _vel in hits:
        # locate the beat interval this hit falls in
        i = int(np.searchsorted(beats, t, side="right") - 1)
        if i < 0 or i >= len(beats) - 1:
            continue
        span = beats[i + 1] - beats[i]
        if span <= 0:
            continue
        frac = (t - beats[i]) / span          # 0..1 within the beat
        sub = round(frac * GRID) / GRID       # snap to sixteenths
        if sub >= 1.0:                        # rounded up into the next beat
            i, sub = i + 1, 0.0
        bar, beat_in_bar = divmod(i, beats_per_bar)
        bars[bar].add((beat_in_bar + 1, sub, note))
    return bars


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def rank_patterns(bars, min_hits=4, similarity=0.6, voices=None):
    """Cluster SIMILAR bar patterns and return a consensus figure for each.

    Exact matching is useless on real playing: a live drummer varies, and
    transcription adds its own noise, so no two bars are ever byte-identical.
    Clustering on Jaccard similarity finds the figure that keeps recurring,
    and the consensus step (keep hits present in most cluster members) drops
    the one-off noise hits while retaining what was consistently played.

    Returns [(consensus_pattern, cluster_size)], largest cluster first.
    """
    patterns = []
    for _bar, pattern in sorted(bars.items()):
        p = {h for h in pattern if voices is None or h[2] in voices}
        if len(p) >= min_hits:
            patterns.append(p)

    unassigned = list(range(len(patterns)))
    clusters = []
    while unassigned:
        # seed with whichever remaining bar has the most neighbours
        neighbours = {
            i: [j for j in unassigned if jaccard(patterns[i], patterns[j]) >= similarity]
            for i in unassigned
        }
        seed = max(unassigned, key=lambda i: (len(neighbours[i]), -i))
        members = neighbours[seed]
        clusters.append(members)
        unassigned = [i for i in unassigned if i not in set(members)]

    out = []
    for members in clusters:
        counts = Counter()
        for i in members:
            counts.update(patterns[i])
        # keep hits played in at least half the cluster's bars
        need = max(1, (len(members) + 1) // 2)
        consensus = frozenset(h for h, c in counts.items() if c >= need)
        if consensus:
            out.append((consensus, len(members)))
    out.sort(key=lambda x: (-x[1], sorted(x[0])))
    return out


def pattern_to_midi(pattern, tempo, path, beats_per_bar=4):
    """Write one quantized bar pattern as a single-bar .mid."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    spb = 60.0 / tempo  # seconds per beat
    for beat, sub, note in sorted(pattern):
        start = ((beat - 1) + sub) * spb
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=note, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(path)


def describe(pattern):
    """Human-readable one-liner for a bar pattern."""
    c = Counter(NAMES.get(n, str(n)) for _b, _s, n in pattern)
    return " ".join(f"{k}x{v}" for k, v in sorted(c.items()))


def selftest():
    """A synthetic 4-bar part where one figure repeats 3x and one bar differs.
    The repeated figure must rank first and be counted 3 times."""
    tempo = 120.0
    spb = 60.0 / tempo
    beats = np.arange(0, 17) * spb  # 16 beats = 4 bars of 4/4

    figure = [(1, 0.0, 36), (2, 0.0, 38), (3, 0.0, 36), (4, 0.0, 38)]
    other = [(1, 0.0, 36), (1, 0.5, 36), (2, 0.0, 38), (4, 0.75, 49)]

    hits = []
    for bar, fig in enumerate([figure, figure, other, figure]):
        for beat, sub, note in fig:
            t = (bar * 4 + (beat - 1) + sub) * spb
            hits.append((t, note, 100))

    bars = quantize_to_bars(hits, beats)
    ranked = rank_patterns(bars, min_hits=4)
    assert ranked, "should find patterns"
    top, count = ranked[0]
    assert count == 3, f"repeated figure should appear 3x, got {count}"
    assert set(top) == set(figure), "top pattern should be the repeated figure"
    assert len(ranked) == 2, f"expected 2 distinct patterns, got {len(ranked)}"
    print(f"OK — top figure recurred {count}x, {len(ranked)} distinct patterns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", help="drums stem (for beat tracking)")
    ap.add_argument("mid", nargs="?", help="transcribed .mid")
    ap.add_argument("-o", "--outdir", default="cells")
    ap.add_argument("-n", "--top", type=int, default=12, help="how many cells to emit")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=2,
                    help="pattern must recur at least this often")
    ap.add_argument("--similarity", type=float, default=0.6,
                    help="0..1 Jaccard threshold for calling two bars the same figure")
    ap.add_argument("--voices", default="all",
                    help="'all', or comma list e.g. kick,snare (skeleton only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.wav and args.mid):
        ap.error("need wav and mid (or --selftest)")

    import pretty_midi

    beats, tempo = beat_grid(args.wav)
    pm = pretty_midi.PrettyMIDI(args.mid)
    hits = [(n.start, n.pitch, n.velocity) for n in pm.instruments[0].notes]

    voices = None
    if args.voices != "all":
        want = {v.strip() for v in args.voices.split(",")}
        voices = {n for n, name in NAMES.items() if name in want}

    bars = quantize_to_bars(hits, beats, args.beats_per_bar)
    ranked = rank_patterns(bars, similarity=args.similarity, voices=voices)
    kept = [(p, c) for p, c in ranked if c >= args.min_count][:args.top]

    print(f"tempo ~{tempo:.1f}bpm | {len(beats)} beats | {len(bars)} bars")
    print(f"{len(ranked)} distinct figures, "
          f"{sum(1 for _, c in ranked if c >= args.min_count)} recur >={args.min_count}x")

    if not kept:
        print("nothing recurred — transcription may be too noisy, or lower --min-count")
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nwriting top {len(kept)} to {args.outdir}/:")
    for rank, (pattern, count) in enumerate(kept, 1):
        path = os.path.join(args.outdir, f"cell_{rank:02d}_x{count}.mid")
        pattern_to_midi(pattern, tempo, path, args.beats_per_bar)
        print(f"  {rank:2d}. recurs {count:3d}x | {len(pattern):2d} hits | {describe(pattern)}")


if __name__ == "__main__":
    main()
