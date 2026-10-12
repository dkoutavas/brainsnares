#!/usr/bin/env bash
# Run every script's --selftest, current pipeline and superseded scripts alike.
# Green means the code is ready to check by ear. It does not mean it works.
#
#   ./selftest.sh              # uses .venv/bin/python
#   PY=python3 ./selftest.sh   # any interpreter with requirements-selftest.txt
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

fail=0
for s in transcribe interpret mine_cells contact_sheet \
         humanize_stats audio2drums stems2midi cells_from_transcription; do
    printf '%-26s ' "$s"
    if out=$("$PY" "$s.py" --selftest 2>&1); then
        echo "${out##*$'\n'}"
    else
        echo "FAIL"
        echo "$out" | sed 's/^/    /'
        fail=1
    fi
done
exit $fail
