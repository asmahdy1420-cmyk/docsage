"""Run the retrieval evaluation from the repository root.

python evals/run_evals.py -k 3 --min-hit-rate 0.85 --min-mrr 0.7
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from docsage.evaluation import main  # noqa: E402

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    defaults = ["--dataset", str(here / "dataset.jsonl"), "--corpus", str(here / "corpus")]
    raise SystemExit(main(defaults + sys.argv[1:]))
