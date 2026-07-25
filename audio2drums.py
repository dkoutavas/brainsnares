#!/usr/bin/env python3
"""Drum transcriber: drums stem (wav) -> MIDI.

Per-band spectral flux. Each drum voice gets its own frequency band and its
own independent onset detection, so simultaneous hits (kick UNDER a crash —
constant in blast beats) all register. The earlier one-window-one-label
classifier lost every kick that shared a moment with a cymbal.

Velocity comes from the band's flux magnitude, so dynamics survive.

Usage:
    python audio2drums.py drums.wav -o out.mid
    python audio2drums.py drums.wav -o out.mid -s 0.6   # more/less sensitive
    python audio2drums.py --selftest
"""
import argparse

import numpy as np

# General MIDI drum map (matches Ugritone / GM defaults)
KICK, SNARE, HAT, CRASH = 36, 38, 42, 49

HOP = 256  # 5.8ms @44.1k — fine enough for blast beats

# (name, midi_note, low_hz, high_hz, sensitivity_scale, min_gap_ms)
# sensitivity_scale biases how eager each band is; cymbals ring long and
# would otherwise smear into a continuous trigger, so they sit tighter.
# min_gap_ms is a refractory period per drum — a physical limit on how fast
# a human can repeat THAT drum. Without it a kick retriggers on its own
# decay tail and invents notes nobody played. Fastest real double-kick is
# ~16ths at 200bpm = 75ms, so 55ms leaves headroom without inventing hits.
BANDS = [
    ("kick",  KICK,   30,   120,   1.0, 55),
    ("snare", SNARE,  180,  1400,  1.0, 45),
    ("hat",   HAT,    7000, 12000, 0.8, 30),
    ("crash", CRASH,  3000, 6000,  0.7, 55),
]


def band_flux(S, freqs, lo, hi):
    """Positive spectral flux summed over one frequency band."""
    sel = (freqs >= lo) & (freqs < hi)
    band = S[sel, :]
    diff = np.diff(band, axis=1, prepend=band[:, :1])
    flux = np.maximum(diff, 0).sum(axis=0)
    # normalise to 0..1 so delta thresholds mean the same thing per band
    peak = flux.max()
    return flux / peak if peak > 0 else flux


def pick_onsets(flux, sensitivity, scale, min_gap_ms, sr):
    """Peak-pick a flux curve into onset frame indices."""
    import librosa

    # higher sensitivity -> lower threshold -> more hits
    delta = max(0.005, 0.09 * (1.0 - sensitivity) / max(scale, 1e-6))
    wait = max(1, int(round(min_gap_ms / 1000 * sr / HOP)))  # refractory
    return librosa.util.peak_pick(
        flux,
        pre_max=3, post_max=3,      # local max over ~35ms
        pre_avg=12, post_avg=12,    # vs local mean over ~140ms
        delta=delta,
        wait=wait,
    )


def transcribe(y, sr, sensitivity=0.5):
    """Return (hits, tempo). hits = list of (time_s, midi_note, velocity)."""
    import librosa

    S = np.abs(librosa.stft(y, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)

    hits = []
    for _name, note, lo, hi, scale, min_gap in BANDS:
        flux = band_flux(S, freqs, lo, hi)
        for f in pick_onsets(flux, sensitivity, scale, min_gap, sr):
            # flux magnitude -> velocity, keeping a musical floor
            vel = int(np.clip(45 + flux[f] * 180, 30, 127))
            hits.append((float(times[f]), note, vel))

    hits.sort(key=lambda h: h[0])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    try:
        tempo = float(librosa.feature.rhythm.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=HOP)[0])
    except AttributeError:  # older librosa
        tempo = float(librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=HOP)[0])
    return hits, tempo


def write_midi(hits, tempo, path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo or 120.0)
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    for t, note, vel in hits:
        drum.notes.append(pretty_midi.Note(
            velocity=vel, pitch=note, start=t, end=t + 0.05))
    pm.instruments.append(drum)
    pm.write(path)


def selftest():
    """Kick and cymbal struck SIMULTANEOUSLY must both be detected — the
    exact case the old single-label classifier failed."""
    import librosa

    rng = np.random.default_rng(0)
    sr = 44100
    n = int(sr * 0.12)
    env = np.exp(-np.linspace(0, 1, n) * 20)
    gap = np.zeros(int(sr * 0.3))

    def lowpass(x, k):
        return np.convolve(x, np.ones(k) / k, mode="same")

    def kick():
        t = np.arange(n) / sr
        return np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40)

    def cymbal():
        z = rng.standard_normal(n)
        return (z - lowpass(z, 6)) * env

    # cymbal is 3x louder — this is what used to mask the kick entirely
    both = kick() + 3.0 * cymbal()
    y = np.concatenate([gap, both, gap, kick(), gap]).astype(np.float32)
    y = (y / np.abs(y).max()).astype(np.float32)

    hits, _ = transcribe(y, sr, sensitivity=0.6)
    voices = {note for _, note, _ in hits}
    assert KICK in voices, "kick under a loud cymbal must still be detected"
    assert HAT in voices or CRASH in voices, "cymbal must be detected"
    print(f"OK — {len(hits)} hits, voices={sorted(voices)} "
          f"(kick survived a 3x louder cymbal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", help="drums stem wav")
    ap.add_argument("-o", "--out", default="drums.mid")
    ap.add_argument("-s", "--sensitivity", type=float, default=0.5,
                    help="0..1, higher = more hits")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.wav:
        ap.error("need a wav (or --selftest)")

    import librosa
    y, sr = librosa.load(args.wav, sr=None, mono=True)
    hits, tempo = transcribe(y, sr, args.sensitivity)
    write_midi(hits, tempo, args.out)

    counts = {}
    for _, note, _ in hits:
        counts[note] = counts.get(note, 0) + 1
    names = {KICK: "kick", SNARE: "snare", HAT: "hat", CRASH: "crash"}
    summary = ", ".join(f"{names.get(k, k)}={v}" for k, v in sorted(counts.items()))
    print(f"{len(hits)} hits @ ~{tempo:.0f}bpm -> {args.out}  ({summary})")


if __name__ == "__main__":
    main()
