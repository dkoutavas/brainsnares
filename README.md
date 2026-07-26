# brainsnares

Turns reference records into drum vocabulary for [drumgen](../drumgen).
Isolate the drums, transcribe them, find the figures that recur, hand them
over as one-bar cells.

Named after Kidcrash, *Snacks*, track 6.

## Why this exists

drumgen generates drum parts for labyrinthine heavy music from hand-authored
rhythmic cells. Authoring that vocabulary by ear is slow, and its author plays
guitar rather than drums. brainsnares is the shortcut into a first draft of
that vocabulary. Point it at records whose drumming you love and it returns
the bar-length figures those drummers actually repeat.

Nothing here generates music. The tool detects onsets, decides which stem a
hit really belongs to, and counts recurring bars. That constraint comes from
drumgen, whose whole premise is that the composer's taste stays human. This
replaces the tedium of transcription while leaving every judgement about
quality to the person listening.

Treat the output as raw material. The ear decides what survives, and a figure
worth keeping gets hand-authored into drumgen's library rather than shipped
from here.

## Status (2026-07-26)

Working and ear-verified. 11 reference tracks processed (Kidcrash *Snacks* x7,
Lord Snow x3, Ruined Families x1), producing 51 cell candidates, of which 49
imported into drumgen. The other two were correctly rejected by drumgen's
validator as pure-cymbal patterns.

Toms, crash-riding, and overall density came out well. Kick and snare
interplay is the weakest part, and cymbal voicing is approximate.

## The pipeline

```
refs/*.flac
  → run_pipeline.sh     demucs (band → drum stem) → larsnet (→ 5 per-drum stems)
  → transcribe.py       stems → notes    (arbitration, dB velocity, tempo map)
  → interpret.py        notes → part     (crash gate, riding, quantize, consensus)
  → mine_cells.py       bars → 1-bar cell candidates
  → contact_sheet.py    cells → audition clips (one per track)
  → drumgen             per-file import, see CLAUDE.md
```

Everything runs on `.venv/bin/python` (3.11, CUDA torch). Every script has a
`--selftest`.

## Setup

Needs Python 3.11, `ffmpeg`, and ideally an NVIDIA GPU. Developed on a 6 GB
RTX 3060, which separates a two-minute track in about five seconds. CPU works
and is slower.

```sh
python3.11 -m venv .venv          # or: uv venv --python 3.11 .venv
.venv/bin/pip install -r requirements.txt

# LarsNet does the per-drum stem separation. Not vendored here, because its
# pretrained weights are CC BY-NC 4.0.
git clone https://github.com/polimi-ispl/larsnet.git
# then download its pretrained models (562 MB) per that repo's README and
# unzip into larsnet/pretrained_larsnet_models/
```

With torchaudio 2.11 or newer you will need one local patch. Its `load()`
routes through torchcodec, which may fail to link against your ffmpeg. In
`larsnet/larsnet.py`, replace the `ta.load(str(x))` call with soundfile:

```python
import soundfile as _sf, numpy as _np
_data, sr_ = _sf.read(str(x), dtype="float32", always_2d=True)
x = torch.from_numpy(_np.ascontiguousarray(_data.T))   # (channels, samples)
```

Then drop reference audio into `refs/` and run `./run_pipeline.sh`.

The selftests assert real behaviour rather than imports. They check that bleed
is dropped while genuine simultaneous hits survive, that consensus repairs a
noisy bar, and that cells land on the grid. Use them to verify the install:

```sh
for s in transcribe interpret mine_cells contact_sheet; do .venv/bin/python $s.py --selftest; done
```

## What was hard

Getting notes out was never the problem. Getting a drummer out was.

### Separation works; assignment did not

Demucs and LarsNet produce genuinely clean per-drum stems. The catch is that
one physical hit bleeds into two or three of them and every stem's detector
fires, giving 2.31 MIDI notes per hit. That plays back as a continuous drum
solo.

The fix is arbitration on self-relative energy. A hit survives only if it sits
within 9 dB of its own stem's typical hit level, since bleed lands 21 to 30 dB
down. Comparing stems against each other cannot work, because they differ by
more than 20 dB intrinsically, so a genuine kick-plus-hihat double looks
exactly like bleed.

### Then it was notes without being a part

Four musical rules, each found by measuring a section the ear flagged.

1. Crash is usually punctuation, but sometimes it is timekeeping. Gating
   crashes to the strongest within 0.9 s destroyed crash-ridden sections. Runs
   of four or more cymbal hits at regular 8ths or quarters carry the pulse, so
   they are exempt from the gate and keep whatever label the wash/discrete
   classifier gave them. Kidcrash rides the crash at 2:16, confirmed by ear,
   so relabeling those hits to ride was wrong.
2. Toms need their own frequency band. Broadband onset detection lost about
   40% of the heavy tom grooves under cymbal wash. Detecting on 60-350 Hz
   spectral flux recovered them.
3. Tempo has to come from the summed stems with `start_bpm=168`. Six of eleven
   tracks had been written at half tempo. Export carries a per-beat tempo map
   so the DAW grid follows the live drummer.
4. Consensus over repeated bars removes what transcription noise adds.
   Phantom and missing hits are uncorrelated with musical structure, while a
   real part restates its figures. Bars at least 0.7 similar to a figure that
   recurs three or more times snap to it, which brings the riff back. Outlier
   bars, including fills, stay untouched.

## What failed, and is not worth retrying

Pattern mining from raw unquantized transcription found zero exact recurrence
across 52 bars. Only quantize plus consensus made figures countable.

Measuring microtiming for drumgen's humanizer constants does not work on this
material. The beat grid is unreliable, with alignment barely beating chance
(0.78 where 1.0 is random), so timing statistics are contaminated. Do not
derive `TIMING_TENDENCIES` from this pipeline. Velocity ordering did survive
(snare and toms most dynamic, hihat less than assumed), but mix compression
confounds it, so treat that as a hint.

Band classification without stem separation is hopeless. In a distorted mix
snare and cymbals fire together 71% of the time, and frequency bands cannot
tell them apart. Separate first, always.

Lo-fi source material defeats the transcription stage. Tested on raw Colombian
skramz (páramo): separation was excellent, note accuracy unusable.
Kidcrash-grade production is roughly the floor.

Batching LarsNet on a 6 GB GPU runs out of memory. Use one track per
invocation, and CPU for anything over about six minutes.

## The method that worked

Every complaint that something sounded wrong turned out to be one missing
musical rule rather than noise, and each was found the same way.

1. The ear names a bar and describes the feel ("child hitting pots").
2. Measure that section: density, voice mix, grid alignment, per window.
3. The anomaly names the rule.
4. Fix, regenerate, ear checks again.

Threshold tuning was never the answer. Twice, a fix that produced no
structural change turned out to be the signal that the whole approach was
wrong.

## Legal

`refs/` holds copyrighted audio. Never commit it, never distribute it. It is
gitignored.

LarsNet weights are CC BY-NC 4.0 and licensed for personal use. Nothing
derived from them belongs in commercial or public output.

Rhythms are not copyrightable, but transcriptions of identifiable songs should
not be published. If a mined figure earns permanence, hand-author it into
drumgen's `cell_library.py`. Authored cells informed by transcription are the
publishable form.

This repository contains code and notes only. No audio, no transcriptions, no
model weights.

MIT. Bring your own records.
