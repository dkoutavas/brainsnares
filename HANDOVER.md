# drumtool — handover (2026-07-25)

Mines drum vocabulary from reference records into MIDI for drumgen
(`~/dev/drumgen`). Measurement and counting only — no generative models.

## State: working, ear-verified on kidcrash_01

User verdict on the v4 interpreted part: toms great, best result so far,
accepted. Known residuals (logged, not fixed): kick/snare interplay not
fully faithful, cymbal voicing approximate.

## Pipeline

```
refs/*.flac
  → run_pipeline.sh      demucs (drum stem) → larsnet (5 per-drum stems)
  → transcribe.py        stems → notes   (arbitration, velocity, tempo map)
  → interpret.py         notes → part    (crash gate, riding, quantize, consensus)
  → mine_cells.py        bars → 1-bar cell candidate .mids
  → drumgen import       cd ~/dev/drumgen && python midi_reader.py <cells dir> --auto-tag
```

Everything runs with `.venv/bin/python` (3.11, CUDA torch). Each script has
`--selftest`.

## Where things are

| what | where |
|---|---|
| 11 interpreted parts (tempo-mapped .mid) | `work/parts/` |
| 51 cell candidates, named `<track>_<rank>_x<recurs>_<bpm>bpm.mid` | `work/cells/` |
| per-bar figures (JSON) | `work/bars/` |
| isolated per-drum stems | `work/larsnet_all/<drum>/` |
| demucs drum stems | `work/drumstems/` |
| source audio (NOT in git) | `refs/` |
| larsnet clone + weights (NOT in git; CC BY-NC — personal use only) | `larsnet/` |

## The rules that made it work (each one ear-verified)

1. **Arbitration** — one hit bleeds into 2-3 stems and every stem fires
   (2.31 notes per physical hit = "continuous drum solo"). Keep a hit in a
   multi-stem 28ms cluster only if within 9 dB of its OWN stem's typical hit
   energy; bleed sits 21-30 dB down. Plus hihat/cymbals dedupe.
2. **Riding ≠ punctuation** — crash gate (strongest within 0.9s) destroyed
   crash-riding sections ("child hitting pots", bar 85+). Cymbal runs at
   regular 8ths/quarters (4+ hits) are timekeeping: protected from the gate,
   label stays whatever the wash/discrete classifier said (Kidcrash rides
   the crash at 2:16 — confirmed by ear).
3. **Toms detect in their own band** (60-350 Hz spectral flux) — broadband
   envelope lost ~40% of the heavy tom grooves under cymbal wash.
4. **Tempo from summed stems, start_bpm=168** — 6 of 11 tracks were written
   at half tempo before this. Export carries a per-beat tempo map.
5. **Consensus snap** — bars ≥0.7 similar to a ≥3×-recurring figure snap to
   that figure (kills phantom/missing-hit noise, restores the riff); outlier
   bars (fills) stay untouched.

## Dead ends — do not retry

- Pattern mining from raw (unquantized) transcription: zero exact recurrence.
- Microtiming stats for humanizer constants: beat grid unreliable on this
  material (alignment barely better than random). Velocity ORDER survived:
  snare/toms most dynamic, hihat less than drumgen's humanizer assumes
  (mix-compression confound — treat as a hint, not a measurement).
- Band-classification without stem separation (lo-fi páramo test): mush.
- Batch larsnet on the 6GB GPU: OOM. One track per invocation; >6 min
  tracks on CPU.

## Tomorrow / next

1. **User ear pass over `work/cells/`** — keep/kill, then import keepers
   into drumgen (`midi_reader.py <dir> --auto-tag`). Start with x13-x17.
2. **Kick/snare synergy residual** — same drill that fixed everything else:
   user names a bar + what it feels like, measure that section (density,
   voice mix, grid alignment per window), find the missing musical rule.
3. Optional: more refs through the pipeline (drop .flac in `refs/`, run
   `./run_pipeline.sh`, then interpret + mine).
4. Optional cleanup: `rm -rf ~/.local/bitwig-studio.bak-6.0.6` (641M,
   6.0.11 confirmed good).

## Legal reminders

- `refs/` is copyrighted audio — never commit, never distribute.
- LarsNet weights are CC BY-NC 4.0 — personal use; nothing derived from
  them ships in drumgen if it goes commercial/public.
- Mined cells: rhythms aren't copyrightable, but if drumgen's cell library
  goes public, author cells by hand — don't publish transcriptions of
  identifiable songs.
