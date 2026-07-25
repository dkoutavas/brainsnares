#!/usr/bin/env python3
"""Transcribe LarsNet's 5 isolated drum stems into one MIDI file.

Each stem contains ONE drum, so there is no classification step and no
band-overlap problem: detect onsets on the kick stem, those are kicks.
This is what the earlier band-splitting heuristic could not do — in a
distorted mix, snare and cymbals share the whole spectrum above ~200Hz.

Usage:
    python stems2midi.py work/larsnet -o out.mid --name drums.wav
    python stems2midi.py --selftest
"""
import argparse
import os

import numpy as np

# stem folder -> (midi note, min gap ms between repeats of THIS drum)
STEMS = {
    "kick":    (36, 55),
    "snare":   (38, 45),
    "toms":    (45, 60),
    "hihat":   (42, 30),
    "cymbals": (49, 55),
}
HOP = 256


def onsets_for_stem(path, sensitivity, min_gap_ms):
    """Onset times + peak strengths for one isolated drum stem."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    if not np.any(y):
        return [], sr
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    peak = env.max()
    if peak <= 0:
        return [], sr
    env = env / peak

    delta = max(0.005, 0.10 * (1.0 - sensitivity))
    wait = max(1, int(round(min_gap_ms / 1000 * sr / HOP)))
    frames = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=12, post_avg=12,
        delta=delta, wait=wait,
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    return [(float(t), float(env[f])) for t, f in zip(times, frames)], sr


def transcribe_stems(stem_dir, filename, sensitivity=0.5):
    """Walk the LarsNet output folders and build one hit list."""
    hits = []
    for stem, (note, gap) in STEMS.items():
        path = os.path.join(stem_dir, stem, filename)
        if not os.path.exists(path):
            continue
        found, _sr = onsets_for_stem(path, sensitivity, gap)
        for t, strength in found:
            vel = int(np.clip(45 + strength * 170, 30, 127))
            hits.append((t, note, vel))
    hits.sort(key=lambda h: h[0])
    return hits


def write_midi(hits, tempo, path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for t, note, vel in hits:
        inst.notes.append(pretty_midi.Note(
            velocity=vel, pitch=note, start=t, end=t + 0.05))
    pm.instruments.append(inst)
    pm.write(path)


def selftest():
    """An isolated stem containing 4 evenly spaced hits must yield exactly 4
    onsets — the property that makes per-stem detection reliable."""
    import soundfile as sf
    import tempfile

    sr = 44100
    n = int(sr * 0.1)
    env = np.exp(-np.linspace(0, 1, n) * 20)
    t = np.arange(n) / sr
    hit = (np.sin(2 * np.pi * 70 * t) * env).astype(np.float32)
    gap = np.zeros(int(sr * 0.4), dtype=np.float32)
    y = np.concatenate([gap] + [x for _ in range(4) for x in (hit, gap)])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.wav")
        sf.write(p, y, sr)
        found, _ = onsets_for_stem(p, 0.5, 55)
    assert len(found) == 4, f"expected exactly 4 onsets, got {len(found)}"
    print(f"OK — isolated stem gave exactly {len(found)} onsets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem_dir", nargs="?", help="LarsNet output dir")
    ap.add_argument("--name", default="drums.wav", help="stem filename inside each folder")
    ap.add_argument("-o", "--out", default="stems.mid")
    ap.add_argument("-s", "--sensitivity", type=float, default=0.5)
    ap.add_argument("--tempo", type=float, default=None, help="override detected tempo")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.stem_dir:
        ap.error("need stem_dir (or --selftest)")

    import librosa

    hits = transcribe_stems(args.stem_dir, args.name, args.sensitivity)

    tempo = args.tempo
    if tempo is None:
        ref = os.path.join(args.stem_dir, "snare", args.name)
        if not os.path.exists(ref):
            ref = os.path.join(args.stem_dir, "kick", args.name)
        y, sr = librosa.load(ref, sr=None, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        try:
            tempo = float(librosa.feature.rhythm.tempo(
                onset_envelope=env, sr=sr, hop_length=HOP)[0])
        except AttributeError:
            tempo = float(librosa.beat.tempo(
                onset_envelope=env, sr=sr, hop_length=HOP)[0])

    write_midi(hits, tempo, args.out)
    counts = {}
    for _, note, _ in hits:
        counts[note] = counts.get(note, 0) + 1
    label = {36: "kick", 38: "snare", 45: "tom", 42: "hat", 49: "cym"}
    summary = ", ".join(f"{label.get(k, k)}={v}" for k, v in sorted(counts.items()))
    print(f"{len(hits)} hits @ ~{tempo:.0f}bpm -> {args.out}  ({summary})")


if __name__ == "__main__":
    main()
