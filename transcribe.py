#!/usr/bin/env python3
"""LarsNet stems -> MIDI, v2. Parameters measured from the actual material
(README, "What was hard"): arbitration thresholds, velocity anchors,
tom/cymbal split rules, and tempo handling all come from measurement on the
Kidcrash / Lord Snow / Ruined Families stems, not guesses.

The four fixes over stems2midi.py:
  1. ARBITRATION — one physical hit bleeds into 2-3 stems; each stem's
     detector fired, giving 2.31 MIDI notes per hit ("continuous drum solo").
     A hit survives only if its energy is within 9 dB of its own stem's
     typical genuine-hit level (bleed sits 21-30 dB down; valley at ~-9).
  2. VELOCITY — mapped from measured per-hit peak dB (40 dB range on snare,
     bimodal ghosts/accents), not a max-normalized envelope.
  3. VOICES — toms split floor/rack by f0; cymbals split crash/ride by wash
     context; hihat marked open only on isolated long-ring hits.
  4. TEMPO — estimated from the summed stems with start_bpm=168 (fixes six
     2:1 octave errors); export carries a per-beat tempo map so DAW playback
     follows the live drummer instead of drifting off a fixed grid.

Usage:
    python transcribe.py work/larsnet_all --name kidcrash_snacks_01_....wav -o out.mid
    python transcribe.py work/larsnet_all --name track.wav -o out.mid --tighten
    python transcribe.py --selftest
"""
import argparse
import os

import numpy as np

HOP = 256
CLUSTER_S = 0.028          # cross-stem chain-cluster window
E_WIN_S = 0.045            # post-onset RMS window for arbitration energy
SELF_GAP_KEEP_DB = -9.0    # non-anchor cluster member survives above this
REF_MIN_SAMPLES = 8
REF_FALLBACK_DB = -30.0

# stem -> (onset min-gap ms, velocity slope per dB, detection band Hz or None)
# toms detect on band-limited spectral flux (60-350Hz, where the fundamentals
# live): their ring + cymbal-wash bleed swamps the broadband envelope, which
# lost ~40% of the heavy tom groove in kidcrash_01 140-206s. Kick/snare
# showed no benefit from banding (tested) — their dropouts were real playing.
STEM_CFG = {
    "kick":    (55, 2.5, None),
    "snare":   (45, 3.0, None),
    "toms":    (45, 2.5, (60, 350)),
    "hihat":   (30, 2.0, None),
    "cymbals": (55, 2.0, None),
}

# MIDI notes (GM / Ugritone)
N_KICK, N_SNARE = 36, 38
N_TOM_FLOOR, N_TOM_RACK = 41, 47
N_HAT_CLOSED, N_HAT_OPEN = 42, 46
N_RIDE, N_CRASH = 51, 49


# ── detection ────────────────────────────────────────────────────────────────

def detect_stem(y, sr, min_gap_ms, sensitivity=0.5, band=None):
    """Onset frame times for one isolated stem. Frame centers, no backtrack
    (measured: backtrack hurts, interpolation doesn't help; player jitter
    dominates detector error). With `band` (lo, hi) Hz, detection runs on
    positive spectral flux inside that band only — for voices whose attacks
    drown in broadband bleed (toms under cymbal wash)."""
    import librosa

    if band is None:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        delta = max(0.005, 0.10 * (1.0 - sensitivity))
    else:
        S = np.abs(librosa.stft(y, hop_length=HOP))
        freqs = librosa.fft_frequencies(sr=sr)
        bandS = S[(freqs >= band[0]) & (freqs < band[1]), :]
        d = np.diff(bandS, axis=1, prepend=bandS[:, :1])
        env = np.maximum(d, 0).sum(axis=0)
        delta = max(0.005, 0.08 * (1.0 - sensitivity))
    peak = env.max()
    if peak <= 0:
        return np.array([])
    env = env / peak
    frames = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=12, post_avg=12,
        delta=delta,
        wait=max(1, int(round(min_gap_ms / 1000 * sr / HOP))),
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=HOP)


def rms_db(y, sr, t, win_s):
    """RMS energy in dB of y[t : t+win_s]."""
    a = int(t * sr)
    b = min(len(y), a + int(win_s * sr))
    if b <= a:
        return -120.0
    seg = y[a:b]
    return float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12))


def peak_db(y, sr, t):
    """Peak amplitude in dB around the onset (-3ms..+25ms) — velocity basis."""
    a = max(0, int((t - 0.003) * sr))
    b = min(len(y), int((t + 0.025) * sr))
    if b <= a:
        return -120.0
    return float(20 * np.log10(np.abs(y[a:b]).max() + 1e-12))


# ── arbitration ──────────────────────────────────────────────────────────────

def arbitrate(events):
    """events: list of dicts {t, stem, e_db, ...}. Returns surviving events.

    Chain-cluster all events within CLUSTER_S; compute each stem's genuine-hit
    reference (median energy of solo hits + cluster-loudest hits); inside each
    multi-stem cluster keep the best self-relative member unconditionally and
    others only within SELF_GAP_KEEP_DB of their own reference; if hihat and
    cymbals both survive one cluster, keep the higher (LarsNet splits single
    cymbal strokes across both stems)."""
    if not events:
        return []
    events = sorted(events, key=lambda e: e["t"])

    clusters, cur = [], [events[0]]
    for e in events[1:]:
        if e["t"] - cur[-1]["t"] <= CLUSTER_S:
            cur.append(e)
        else:
            clusters.append(cur)
            cur = [e]
    clusters.append(cur)

    # same-stem duplicates inside a cluster -> keep loudest
    for i, c in enumerate(clusters):
        best = {}
        for e in c:
            k = e["stem"]
            if k not in best or e["e_db"] > best[k]["e_db"]:
                best[k] = e
        clusters[i] = list(best.values())

    # per-stem reference: solo onsets + cluster-loudest onsets
    pool = {s: [] for s in STEM_CFG}
    for c in clusters:
        if len(c) == 1:
            pool[c[0]["stem"]].append(c[0]["e_db"])
        else:
            loud = max(c, key=lambda e: e["e_db"])
            pool[loud["stem"]].append(loud["e_db"])
    ref = {s: (float(np.median(v)) if len(v) >= REF_MIN_SAMPLES else REF_FALLBACK_DB)
           for s, v in pool.items()}

    out = []
    for c in clusters:
        if len(c) == 1:
            out.append(c[0])
            continue
        for e in c:
            e["sg"] = e["e_db"] - ref[e["stem"]]
        anchor = max(c, key=lambda e: e["sg"])
        keep = [e for e in c if e is anchor or e["sg"] >= SELF_GAP_KEEP_DB]
        hh = [e for e in keep if e["stem"] == "hihat"]
        cy = [e for e in keep if e["stem"] == "cymbals"]
        if hh and cy:  # one physical cymbal stroke split across two stems
            drop = hh[0] if hh[0]["sg"] < cy[0]["sg"] else cy[0]
            keep = [e for e in keep if e is not drop]
        out.extend(keep)
    return sorted(out, key=lambda e: e["t"])


# ── voice refinement ─────────────────────────────────────────────────────────

def split_toms(events, y, sr):
    """floor vs rack by autocorrelation f0 of an 80ms lowpassed post-onset
    window; GMM valley per track when enough hits, else 110 Hz."""
    from scipy.signal import butter, sosfilt

    toms = [e for e in events if e["stem"] == "toms"]
    if not toms:
        return
    sos = butter(4, 500, btype="low", fs=sr, output="sos")
    f0s = {}
    for e in toms:
        a = int(e["t"] * sr)
        b = min(len(y), a + int(0.080 * sr))
        seg = sosfilt(sos, y[a:b])
        if len(seg) < 200 or not np.any(seg):
            continue
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(sr / 350), int(sr / 50)   # 50-350 Hz
        if hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        f0 = sr / lag
        if 60 <= f0 <= 250:
            f0s[id(e)] = f0
    thr = 110.0
    vals = np.array(list(f0s.values()))
    if len(vals) >= 15:
        try:
            from sklearn.mixture import GaussianMixture
            g = GaussianMixture(2, random_state=0).fit(np.log(vals).reshape(-1, 1))
            m = np.sort(np.exp(g.means_.ravel()))
            thr = float(np.sqrt(m[0] * m[1]))  # geometric midpoint of modes
        except Exception:
            pass
    for e in toms:
        f0 = f0s.get(id(e))
        e["note"] = N_TOM_FLOOR if (f0 is not None and f0 < thr) else N_TOM_RACK


def split_cymbals(events, y, sr):
    """crash (wash context) vs ride (discrete decaying articulation).
    pre_db: cymbal bed level just before the onset, relative to track median;
    floor_rel: how far the envelope dips after the hit before the next one."""
    import librosa

    cyms = [e for e in events if e["stem"] == "cymbals"]
    if not cyms:
        return
    env = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024,
                                            hop_length=HOP)[0] + 1e-12)
    tframe = np.arange(len(env)) * HOP / sr
    active = env[env > env.max() - 60]
    med = float(np.median(active)) if len(active) else -60.0
    times = [e["t"] for e in cyms]
    for i, e in enumerate(cyms):
        f = int(e["t"] * sr / HOP)
        pre = env[max(0, f - int(0.050 * sr / HOP)):f]
        pre_db = (float(np.median(pre)) - med) if len(pre) else 0.0
        t_next = times[i + 1] if i + 1 < len(cyms) else e["t"] + 1.0
        a = f + int(0.040 * sr / HOP)
        b = min(len(env), int(t_next * sr / HOP))
        floor_rel = (float(env[a:b].min()) - float(env[f])) if b > a else 0.0
        e["note"] = N_RIDE if (pre_db <= 0 and floor_rel < -6.0) else N_CRASH


def split_hihat(events, y, sr):
    """open only when isolated AND ringing (not separable in dense streams —
    measured unimodal; do not pretend otherwise)."""
    import librosa

    hats = [e for e in events if e["stem"] == "hihat"]
    if not hats:
        return
    env = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024,
                                            hop_length=HOP)[0] + 1e-12)
    times = [e["t"] for e in hats]
    for i, e in enumerate(hats):
        e["note"] = N_HAT_CLOSED
        ioi = (times[i + 1] - e["t"]) if i + 1 < len(hats) else 10.0
        if ioi <= 0.45:
            continue
        f = int(e["t"] * sr / HOP)
        if f >= len(env):
            continue
        pk = env[f]
        seg = env[f:min(len(env), f + int(ioi * sr / HOP))]
        below = np.nonzero(seg < pk - 20)[0]
        d20 = (below[0] * HOP / sr) if len(below) else ioi
        if d20 > 0.25:
            e["note"] = N_HAT_OPEN


# ── velocity ─────────────────────────────────────────────────────────────────

def velocity_from_peak(p_db, p95_db, slope):
    """Peak dB -> MIDI velocity, anchored so the stem's p95 peak lands on 120.
    Slope per stem (3.0 snare, 2.5 kick/toms, 2.0 cymbals) keeps ghost/accent
    bimodality across ~40 dB of snare range. Skews accent-heavy by design."""
    return int(np.clip(round(120 + slope * (p_db - p95_db)), 30, 127))


# ── tempo + export ───────────────────────────────────────────────────────────

def track_tempo(stems, sr):
    """Beat times from the SUMMED stems, start_bpm=168 (fixes the 2:1 octave
    errors that put Kidcrash at 83bpm). Returns (beat_times, bpm)."""
    import librosa

    n = max(len(y) for y in stems.values())
    mix = np.zeros(n, dtype=np.float64)
    for y in stems.values():
        mix[:len(y)] += y
    env = librosa.onset.onset_strength(y=mix, sr=sr, hop_length=HOP)

    def run(start):
        _t, beats = librosa.beat.beat_track(
            onset_envelope=env, sr=sr, hop_length=HOP,
            start_bpm=start, trim=False, units="time")
        if len(beats) < 8:
            return None
        ioi = np.diff(beats)
        bpm = 60.0 / float(np.median(ioi))
        spread = (np.percentile(ioi, 90) - np.percentile(ioi, 10)) / np.median(ioi)
        return beats, bpm, spread

    a = run(168)
    if a and 140 <= a[1] <= 200:
        return a[0], a[1]
    b = run(120)
    cands = [c for c in (a, b) if c]
    if not cands:
        raise SystemExit("tempo: fewer than 8 beats found on the summed stems; "
                         "is this a whole track?")
    best = min(cands, key=lambda c: c[2])
    return best[0], best[1]


def write_tempo_map_midi(events, beats, bpm, path, tighten=False):
    """MIDI with one set_tempo per beat so the DAW grid follows the drummer.
    Note ticks are positions in BEAT space (t mapped through the beat list)."""
    import mido

    PPQ = 480
    ioi = np.diff(beats)
    med = float(np.median(ioi))
    # 5-beat median smooth, clamped to [0.7, 1.43]x median
    smooth = np.copy(ioi)
    for i in range(len(ioi)):
        smooth[i] = np.median(ioi[max(0, i - 2):i + 3])
    smooth = np.clip(smooth, 0.7 * med, 1.43 * med)

    def to_beat_pos(t):
        i = int(np.searchsorted(beats, t, side="right") - 1)
        if i < 0:
            return max(0.0, (t - beats[0]) / smooth[0] + 0.0)
        if i >= len(beats) - 1:
            return (len(beats) - 1) + (t - beats[-1]) / smooth[-1]
        return i + (t - beats[i]) / (beats[i + 1] - beats[i])

    notes = []
    for e in events:
        bp = to_beat_pos(e["t"])
        if tighten:
            # snap 50% toward the nearest 16th when within 30ms of it
            near = round(bp * 4) / 4
            if abs(bp - near) * smooth[min(int(bp), len(smooth) - 1)] <= 0.030:
                bp = bp + 0.5 * (near - bp)
        notes.append((bp, e["note"], e["vel"]))
    notes.sort()

    mid = mido.MidiFile(ticks_per_beat=PPQ)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)

    msgs = []
    for i, s in enumerate(smooth):
        msgs.append((float(i), "tempo", int(s * 1e6)))
    for bp, note, vel in notes:
        tick = bp
        msgs.append((tick, "on", (note, vel)))
        msgs.append((tick + 0.10, "off", (note, 0)))
    msgs.sort(key=lambda m: (m[0], 0 if m[1] == "tempo" else 1))

    last = 0
    for pos, kind, val in msgs:
        tick = max(0, int(round(pos * PPQ)))
        delta = tick - last
        last = tick
        if kind == "tempo":
            tr.append(mido.MetaMessage("set_tempo", tempo=val, time=delta))
        elif kind == "on":
            tr.append(mido.Message("note_on", channel=9, note=val[0],
                                   velocity=val[1], time=delta))
        else:
            tr.append(mido.Message("note_off", channel=9, note=val[0],
                                   velocity=0, time=delta))
    mid.save(path)


# ── pipeline ─────────────────────────────────────────────────────────────────

def transcribe(stem_dir, name, sensitivity=0.5):
    import librosa

    stems, srs = {}, set()
    for stem in STEM_CFG:
        p = os.path.join(stem_dir, stem, name)
        if os.path.exists(p):
            y, sr = librosa.load(p, sr=None, mono=True)
            stems[stem] = y
            srs.add(sr)
    if not stems:
        raise SystemExit(f"no stems for {name} under {stem_dir}")
    sr = srs.pop()

    events = []
    for stem, y in stems.items():
        gap, _slope, band = STEM_CFG[stem]
        for t in detect_stem(y, sr, gap, sensitivity, band=band):
            events.append({
                "t": float(t), "stem": stem,
                "e_db": rms_db(y, sr, t, E_WIN_S),
                "p_db": peak_db(y, sr, t),
            })

    kept = arbitrate(events)

    # velocity: anchored to each stem's p95 peak level, slope per stem
    for stem in stems:
        se = [e for e in kept if e["stem"] == stem]
        if not se:
            continue
        p95 = float(np.percentile([e["p_db"] for e in se], 95))
        slope = STEM_CFG[stem][1]
        for e in se:
            e["vel"] = velocity_from_peak(e["p_db"], p95, slope)

    # base notes, then refinements
    base = {"kick": N_KICK, "snare": N_SNARE, "toms": N_TOM_RACK,
            "hihat": N_HAT_CLOSED, "cymbals": N_CRASH}
    for e in kept:
        e["note"] = base[e["stem"]]
    if "toms" in stems:
        split_toms(kept, stems["toms"], sr)
    if "cymbals" in stems:
        split_cymbals(kept, stems["cymbals"], sr)
    if "hihat" in stems:
        split_hihat(kept, stems["hihat"], sr)

    beats, bpm = track_tempo(stems, sr)
    return kept, beats, bpm


def selftest():
    """Arbitration: a loud hit in stem A with a 25dB-down echo in stem B must
    keep A and drop B, while an honest simultaneous hit in both survives."""
    ev = []
    # 12 solo snare hits at ~ -15 dB build the snare reference
    for i in range(12):
        ev.append({"t": i * 0.5, "stem": "snare", "e_db": -15.0, "p_db": -8})
    # 12 solo hihat hits at ~ -36 dB build the hihat reference
    for i in range(12):
        ev.append({"t": i * 0.5 + 0.25, "stem": "hihat", "e_db": -36.0, "p_db": -30})
    # bleed case: snare hit with hihat echo 25 dB under hihat's own reference
    ev.append({"t": 100.0, "stem": "snare", "e_db": -15.0, "p_db": -8})
    ev.append({"t": 100.01, "stem": "hihat", "e_db": -61.0, "p_db": -55})
    # genuine double: snare + hihat both at their own full level
    ev.append({"t": 200.0, "stem": "snare", "e_db": -15.0, "p_db": -8})
    ev.append({"t": 200.01, "stem": "hihat", "e_db": -36.0, "p_db": -30})

    # 12 solo cymbal hits at ~ -20 dB build the cymbals reference
    for i in range(12):
        ev.append({"t": 300.0 + i * 0.5, "stem": "cymbals", "e_db": -20.0, "p_db": -12})
    # one cymbal stroke LarsNet split across both stems, each within 9 dB of
    # its own reference: keep only the one closer to its reference (hihat)
    ev.append({"t": 400.0, "stem": "hihat", "e_db": -36.0, "p_db": -30})
    ev.append({"t": 400.01, "stem": "cymbals", "e_db": -24.0, "p_db": -16})

    out = arbitrate(ev)
    at100 = [e["stem"] for e in out if abs(e["t"] - 100.0) < 0.05]
    at200 = [e["stem"] for e in out if abs(e["t"] - 200.0) < 0.05]
    at400 = [e["stem"] for e in out if abs(e["t"] - 400.0) < 0.05]
    assert at100 == ["snare"], f"bleed must be dropped, got {at100}"
    assert sorted(at200) == ["hihat", "snare"], f"genuine double must survive, got {at200}"
    assert at400 == ["hihat"], f"hihat+cymbals split stroke must collapse to one, got {at400}"

    # velocity: p95 peak -> 120, 10 dB under on snare -> 90, floor and ceiling clip
    assert velocity_from_peak(-8.0, -8.0, 3.0) == 120
    assert velocity_from_peak(-18.0, -8.0, 3.0) == 90
    assert velocity_from_peak(-60.0, -8.0, 3.0) == 30, "ghosts floor at 30"
    assert velocity_from_peak(0.0, -8.0, 3.0) == 127, "ceiling at 127"

    # tempo-map export: a hit on beat 3 at a steady 120 bpm lands at tick 3*PPQ,
    # one set_tempo per beat interval, and --tighten pulls a 10 ms-late hit
    # halfway back to the sixteenth
    import tempfile
    import mido
    beats = np.arange(16) * 0.5
    hits = [{"t": 1.5, "note": N_KICK, "vel": 100},
            {"t": 1.5 + 0.010, "note": N_SNARE, "vel": 100}]

    def on_ticks(path):
        t, out = 0, {}
        for m in mido.MidiFile(path).tracks[0]:
            t += m.time
            if m.type == "note_on":
                out[m.note] = t
            elif m.type == "set_tempo":
                out.setdefault("tempos", []).append(m.tempo)
        return out

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_tempo_map_midi(hits, beats, 120.0, p)
        got = on_ticks(p)
        assert got[N_KICK] == 3 * 480, f"beat 3 must be tick 1440, got {got[N_KICK]}"
        assert got[N_SNARE] == 1450, f"10 ms late at 120 bpm = 3.02 beats, got {got[N_SNARE]}"
        assert len(got["tempos"]) == len(beats) - 1 and set(got["tempos"]) == {500000}
        write_tempo_map_midi(hits, beats, 120.0, p, tighten=True)
        assert on_ticks(p)[N_SNARE] == 1445, "tighten snaps halfway to the sixteenth"

    print("OK — bleed dropped, genuine double kept, split cymbal collapsed, "
          "velocity map and tempo-map export verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem_dir", nargs="?")
    ap.add_argument("--name", help="stem filename inside each folder")
    ap.add_argument("-o", "--out", default="out.mid")
    ap.add_argument("-s", "--sensitivity", type=float, default=0.5)
    ap.add_argument("--tighten", action="store_true",
                    help="snap hits 50%% toward nearest 16th when within 30ms")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.stem_dir and args.name):
        ap.error("need stem_dir and --name (or --selftest)")

    kept, beats, bpm = transcribe(args.stem_dir, args.name, args.sensitivity)
    write_tempo_map_midi(kept, beats, bpm, args.out, tighten=args.tighten)

    dur = kept[-1]["t"] - kept[0]["t"] if len(kept) > 1 else 1
    names = {N_KICK: "kick", N_SNARE: "snare", N_TOM_FLOOR: "ftom",
             N_TOM_RACK: "rtom", N_HAT_CLOSED: "hat", N_HAT_OPEN: "ohat",
             N_RIDE: "ride", N_CRASH: "crash"}
    counts = {}
    for e in kept:
        counts[e["note"]] = counts.get(e["note"], 0) + 1
    vels = [e["vel"] for e in kept]
    summary = ", ".join(f"{names.get(k, k)}={v}" for k, v in sorted(counts.items()))
    print(f"{len(kept)} notes @ {bpm:.0f}bpm ({len(kept)/dur:.1f}/s) -> {args.out}")
    print(f"  {summary}")
    print(f"  velocity: min {min(vels)} / median {int(np.median(vels))} / max {max(vels)}")


if __name__ == "__main__":
    main()
