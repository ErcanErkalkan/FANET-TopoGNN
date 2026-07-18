from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.external_claims import lint_external_claim_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject external-data claims that cross declared evidence boundaries.")
    parser.add_argument("paths", type=Path, nargs="*", default=[ROOT / "paper" / "main.tex"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    violations = [item for path in args.paths for item in lint_external_claim_path(path)]
    if args.json:
        print(json.dumps({"status": "fail" if violations else "pass", "violations": violations}, indent=2))
    elif violations:
        for item in violations:
            print(f"[FAIL] {item['rule_id']} {item['source']}:{item['line']}: {item['text']}")
    else:
        print("[PASS] external claim boundaries")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
