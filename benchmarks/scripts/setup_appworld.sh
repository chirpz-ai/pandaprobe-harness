#!/usr/bin/env bash
# Provision AppWorld in an ISOLATED venv (it pins pydantic<2, irreconcilable with
# the pandabench core). pandabench then drives it over HTTP; it is never imported
# in-process. Prints the two env vars to export for real AppWorld runs.
set -euo pipefail

AWVENV="${PANDABENCH_APPWORLD_VENV:-$HOME/.pandabench/awenv}"
AWROOT="${APPWORLD_ROOT:-$HOME/.pandabench/appworld}"

echo ">> creating isolated AppWorld venv at $AWVENV (python 3.12)"
uv venv "$AWVENV" --python 3.12
uv pip install --python "$AWVENV/bin/python" "appworld==0.1.3.post1"

echo ">> appworld install (decrypt apps + tests)"
"$AWVENV/bin/appworld" install

echo ">> appworld download data --root $AWROOT (~183 MB)"
mkdir -p "$AWROOT"
APPWORLD_ROOT="$AWROOT" "$AWVENV/bin/appworld" download data --root "$AWROOT"

cat <<EOF

AppWorld isolated env ready. Export these (or add to benchmarks/.env) before
real AppWorld runs:

  export PANDABENCH_APPWORLD_PYTHON=$AWVENV/bin/python
  export APPWORLD_ROOT=$AWROOT

EOF
