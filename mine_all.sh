#!/usr/bin/env bash
# Steps 3 and 4 over every separated track: interpret each one into a part
# and its per-bar figures, then mine cells and build contact sheets from all
# of them at once. Resumable: a track with a bars.json is skipped unless
# FORCE=1 is set.
#
#   ./mine_all.sh
#   FORCE=1 ./mine_all.sh      # re-interpret everything (after a code change)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
STEMS=work/larsnet_all

mkdir -p work/parts work/bars
for k in "$STEMS"/kick/*.wav; do
    [ -e "$k" ] || { echo "no stems under $STEMS/kick/ (run ./run_pipeline.sh first)" >&2; exit 1; }
    name=$(basename "$k")
    slug=${name%.wav}
    if [ -f "work/bars/$slug.json" ] && [ -z "${FORCE:-}" ]; then
        echo "skip (done): $slug"
        continue
    fi
    echo ">> interpret: $slug"
    "$PY" interpret.py "$STEMS" --name "$name" \
        -o "work/parts/$slug.mid" --dump-bars "work/bars/$slug.json"
done

echo
echo ">> mine cells"
"$PY" mine_cells.py "work/bars/*.json" -o work/cells
echo
echo ">> contact sheets"
"$PY" contact_sheet.py "work/bars/*.json" -o work/sheets
