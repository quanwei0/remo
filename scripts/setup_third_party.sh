#!/bin/bash
# Install the core and (for FinanceGym) the vendored official FinanceHarness into the active environment. Idempotent.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
python -m pip install -q -e . && echo "remo installed into $(python -c 'import sys; print(sys.prefix)')"
if [ "${1:-}" = "financegym" ]; then
  python -m pip install -q -e third_party/finance_harness && echo "finance_harness installed (FinanceGym only)"
fi
