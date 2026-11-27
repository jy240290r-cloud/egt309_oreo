#!/usr/bin/env bash
# Run the predictive maintenance pipeline from the project root.
# Extra arguments are passed on to main.py, e.g. ./run.sh --models random_forest --no-tune
set -euo pipefail

cd "$(dirname "$0")"
python src/main.py --config src/config/config.yaml "$@"
