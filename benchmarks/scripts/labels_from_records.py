#!/usr/bin/env python
"""Convert a run's records.jsonl into a pandaprobe-harness-calibrate label file.

    python scripts/labels_from_records.py <records.jsonl> <labels.json> \
        --benchmark appworld [--phase learning] [--arm harness]

Thin CLI wrapper over ``pandabench.checkpoints.records_to_labels`` (Checkpoint 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pandabench.checkpoints import records_to_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--phase", default="learning")
    parser.add_argument("--arm", default="harness")
    args = parser.parse_args()
    n = records_to_labels(
        args.records, args.labels, benchmark=args.benchmark, phase=args.phase, arm=args.arm
    )
    print(f"wrote {n} labels to {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
