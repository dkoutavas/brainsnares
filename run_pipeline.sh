#!/usr/bin/env bash
# refs/*.flac -> demucs (drum stem) -> larsnet (5 per-drum stems)
#
# Two stages because they do different jobs: demucs pulls drums out of the
# band, larsnet splits that drum mix into individual drums. Both run on GPU.
set -euo pipefail

cd "$(dirname "$0")"
PY=.venv/bin/python
REFS=refs
DRUMS=work/drumstems
OUT=work/larsnet_all

mkdir -p "$DRUMS" "$OUT"

# stage 1: band mix -> drum stem, one per track, named by slug
for f in "$REFS"/*.flac "$REFS"/*.mp3 "$REFS"/*.wav; do
    [ -e "$f" ] || continue
    base=$(basename "${f%.*}")
    # slug: lowercase, non-alnum -> underscore, squeeze, trim
    slug=$(echo "$base" | tr '[:upper:]' '[:lower:]' \
           | sed -e 's/[^a-z0-9]\+/_/g' -e 's/^_//' -e 's/_$//')
    if [ -f "$DRUMS/$slug.wav" ]; then
        echo "skip (done): $slug"
        continue
    fi
    echo ">> demucs: $slug"
    $PY -m demucs -n htdemucs --two-stems=drums -o work/demucs "$f" >/dev/null 2>&1
    src="work/demucs/htdemucs/$base/drums.wav"
    if [ -f "$src" ]; then
        cp "$src" "$DRUMS/$slug.wav"
    else
        echo "   !! no drum stem produced for $base" >&2
    fi
done

echo
echo ">> larsnet: splitting $(ls -1 "$DRUMS" | wc -l) drum stems into 5 parts each"
# One track per invocation: larsnet loads whole tracks onto the GPU, and a
# batch of 11 blows past 6GB VRAM. Serial keeps peak memory at one track.
mkdir -p work/_one
for s in "$DRUMS"/*.wav; do
    [ -e "$s" ] || continue
    name=$(basename "$s")
    if [ -f "$OUT/kick/$name" ]; then
        echo "   skip (done): $name"
        continue
    fi
    echo "   $name"
    rm -f work/_one/*.wav
    cp "$s" work/_one/
    (cd larsnet && ../$PY separate.py -i ../work/_one -o "../$OUT" -d cuda:0 \
        >/dev/null 2>>../work/larsnet_errors.log) || echo "   !! failed: $name" >&2
done
rm -rf work/_one

echo
echo "stems per drum:"
for d in kick snare toms hihat cymbals; do
    n=$(ls -1 "$OUT/$d" 2>/dev/null | wc -l)
    echo "  $d: $n"
done
