#!/usr/bin/env bash
# Execute notebooks top to bottom in a clean kernel and fail on the first error.
#
#   scripts/run_notebooks.sh                 # every notebook
#   scripts/run_notebooks.sh "M2 - Machine Learning/02-model-selection.ipynb"
#
# Executed copies are written to a temporary directory so the committed notebooks
# (which ship with their reviewed outputs) are never modified.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
TIMEOUT="${TIMEOUT:-1200}"

if [ "$#" -gt 0 ]; then
  notebooks=("$@")
else
  notebooks=()
  while IFS= read -r -d '' nb; do notebooks+=("$nb"); done \
    < <(find . -path ./.git -prune -o -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 | sort -z)
fi

status=0
for nb in "${notebooks[@]}"; do
  name="$(basename "$nb")"
  printf '\n==> %s\n' "$nb"
  if jupyter nbconvert --to notebook --execute \
       --ExecutePreprocessor.timeout="$TIMEOUT" \
       --output-dir "$OUT_DIR" --output "$name" "$nb"; then
    printf 'ok    %s\n' "$nb"
  else
    printf 'FAIL  %s\n' "$nb"
    status=1
  fi
done

echo
echo "Executed copies: $OUT_DIR"
exit "$status"
