# CLAUDE.md

Guidance for Claude Code working in this repository.

Read `README.md` first for purpose, the problems solved, and what failed. This
file is the mechanical reference: commands, architecture, parameters, gotchas.

## Project overview

brainsnares mines drum vocabulary from reference records for drumgen
(`~/dev/drumgen`), whose `CLAUDE.md` carries a matching section called "Mined
cells from reference records". Audio goes in, one-bar cell candidates come out.

No model generates notes here. The tool measures and counts, and that
constraint is inherited from drumgen's stated beliefs rather than being a
preference about style. Expect this to stay an internal tool for drumgen's
library-building phase.

Parked since October 2026, working and tested. The README's "Status" section
says why and how to resume. A plugin or DAW integration was considered and
set aside: the friction is upstream (no song needing drums), not in the
tooling, and LarsNet's CC BY-NC weights rule out distributing one anyway.

## Commands

```bash
# Setup already exists: .venv (Python 3.11, CUDA torch). Use it explicitly.
PY=.venv/bin/python

# 0. Drop reference audio into refs/ (any .flac/.mp3/.wav)

# 1. Separate: band -> drum stem -> 5 per-drum stems. Resumable, skips done work.
./run_pipeline.sh

# 2. Transcribe one track: stems -> notes
$PY transcribe.py work/larsnet_all --name <track>.wav -o out.mid
$PY transcribe.py work/larsnet_all --name <track>.wav -o out.mid --tighten

# 3. Interpret: notes -> musical part. This is what makes it sound composed.
$PY interpret.py work/larsnet_all --name <track>.wav -o part.mid --dump-bars bars.json
$PY interpret.py ... --no-consensus        # skip riff repair, for A/B diagnosis

# 4. Mine cells + build audition sheets
$PY mine_cells.py "work/bars/*.json" -o work/cells
$PY contact_sheet.py "work/bars/*.json" -o work/sheets

# 3 + 4 over every separated track. Resumable; FORCE=1 re-interprets all.
./mine_all.sh

# Every script self-tests. The runner covers all eight, legacy ones included.
./selftest.sh
$PY transcribe.py --selftest        # or one at a time
```

CI (`.github/workflows/selftest.yml`) runs `./selftest.sh` on every push with
`requirements-selftest.txt`, a CPU-only subset with no torch, demucs or
LarsNet. Separation is never exercised by tests; only the arithmetic after it.

## Architecture

| file | role |
|---|---|
| `run_pipeline.sh` | demucs then larsnet, per track, resumable |
| `mine_all.sh` | interpret every track, then mine cells and sheets, resumable |
| `selftest.sh` | runs every script's `--selftest`, used by CI |
| `transcribe.py` | stems to notes: detection, arbitration, velocity, tempo, MIDI |
| `interpret.py` | notes to part: crash gate, riding, quantize, bar consensus |
| `mine_cells.py` | bars to recurring figures to one-bar cell `.mid`s |
| `contact_sheet.py` | cells to one audition clip per track, with markers |
| `humanize_stats.py` | feel measurement, superseded (see README failures) |
| `audio2drums.py`, `stems2midi.py`, `cells_from_transcription.py` | earlier approaches, kept as the record of what did not work |

The superseded scripts say so in their docstrings and still self-test, so a
dependency change that breaks them shows up in CI rather than silently.

Data lives under `work/`, all of it gitignored and regenerable: `drumstems/`,
`larsnet_all/<drum>/`, `parts/`, `bars/`, `cells/`, `sheets/`.

## Key parameters

Every value below came from measuring the actual reference material. Changing
any of them means re-verifying by ear.

### Arbitration (`transcribe.py`)

Cross-stem cluster window is 28 ms. A
non-anchor hit survives only if it falls within 9 dB of its own stem's
reference, computed as the median energy of solo hits plus cluster-loudest
hits, requiring at least 8 samples and defaulting to -30 dBFS. The
cluster-strongest member always survives. When hihat and cymbals both survive
one cluster, keep the higher of the two, because LarsNet splits single cymbal
strokes across both stems.

### Velocity

`clip(120 + slope * (peak_dB - p95_of_stem), 30, 127)`, with
slope 3.0 for snare, 2.5 for kick and toms, 2.0 for hihat and cymbals. This
preserves ghost and accent bimodality across roughly 40 dB of range on snare.
It skews accent-heavy by design.

### Detection bands

Toms use 60-350 Hz spectral flux, everything else is
broadband. Minimum gaps are 55 ms kick, 45 snare, 45 toms, 30 hihat, 55
cymbals.

### Tempo

Run `beat_track` on the summed stems with `start_bpm=168` and
accept a result between 140 and 200. Otherwise re-run at 120 and take the
steadier candidate.

### Crash gate (`interpret.py`)

The strongest crash within 0.9 s wins.
Riding runs are exempt and keep their existing crash or ride label. A riding
run is four or more cymbal hits at regular quarters or 8ths within 25%.

### Consensus

Cluster bars at Jaccard 0.6 or above. A bar at least 0.7
similar to a figure recurring three or more times snaps to that figure's
consensus, defined as the hits present in at least half the cluster. Outliers
are left alone.

## Handing cells to drumgen

Cells are already correct-BPM, GM/Ugritone-mapped MIDI (36/38/41/47/42/46/51/49),
so they can be dragged straight into a DAW with no import step.

For drumgen import, go per file with an explicit `--name`:

```bash
cd ~/dev/drumgen
.venv/bin/python midi_reader.py ~/dev/brainsnares/work/cells/<file>.mid \
    --name kc_mountains_01_x13 --auto-tag
```

Avoid bulk-importing the directory. `_import_directory` accepts no `--name` or
`--tags`, which produces doubled `_<bpm>bpm_<bpm>bpm` names and no provenance,
and mass-importing unaudited cells scatters them through curated `TAG_TO_POOLS`
pools. `auto_tag_cell` also discards caller-supplied `--tags`, so provenance
tags such as `mined` and the band name get patched into the `user_cells/` JSON
after import.

Cell filenames carry provenance as
`<band>_<track>_<rank>_x<recurrence>_<bpm>bpm.mid`. A higher `x` value means
the figure carried more of the original song.

## Working agreements

- Nothing is verified until it has been heard. Green selftests mean the code
  is ready to check. Say that plainly instead of claiming it works.
- The ear is the gate. When output is wrong, ask for a bar number and a
  description of the feel, then measure that window. Do not tune thresholds
  hoping for improvement.
- A fix that produces no structural change means the theory is wrong. That
  happened twice here, and both times the approach was at fault rather than
  the parameter.
- Every non-trivial script keeps one runnable `--selftest` asserting the
  behaviour that matters, such as bleed being dropped while genuine doubles
  survive. A rule written into this file or the README should have an
  assertion behind it; the arbitration, velocity, tempo-map, crash gate,
  riding, quantize and consensus rules all do.
- Deliberate shortcuts get a `ponytail:` comment naming the ceiling and the
  upgrade path.
- Never commit `refs/`, `work/`, or `larsnet/`. Those hold copyrighted audio,
  derived data, and CC BY-NC weights. See the README's legal section.
