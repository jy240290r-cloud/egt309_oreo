#!/usr/bin/env bash
# Run the predictive maintenance pipeline from the project root.
# Extra arguments are passed on to main.py, e.g. ./run.sh --models random_forest --no-tune
#
# Works on Linux, macOS, Docker and Windows (Git Bash). Set PYTHON to force a
# specific interpreter, e.g.  PYTHON=python3.11 ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

# Pick the first Python that exists AND has the project's packages installed.
# (Names differ: Linux/Docker usually have python3, Windows has python or py.
#  On Windows, PowerShell's `bash` may start WSL, whose Python lacks the packages.)
find_python() {
    local candidate
    for candidate in "${PYTHON:-}" python3 python py; do
        [ -z "$candidate" ] && continue
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c "import pandas, sklearn, yaml, joblib, matplotlib" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! PY="$(find_python)"; then
    echo "ERROR: no Python with the required packages was found." >&2
    echo "  1. Install them:  pip install -r requirements.txt" >&2
    echo "  2. On Windows, run this script from Git Bash (not PowerShell's bash, which may be WSL)," >&2
    echo "     or run the pipeline directly:  python src/main.py" >&2
    exit 1
fi

echo "Using $(command -v "$PY") ($("$PY" --version 2>&1))"
"$PY" src/main.py --config src/config/config.yaml "$@"