from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.evaluation.ranking_diagnostics import (  # noqa: E402
    RankingDiagnosticsConfig,
    run_ranking_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ranking diagnostics from embeddings.npz.")
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    run_ranking_diagnostics(
        RankingDiagnosticsConfig(
            embeddings_path=args.embeddings_path,
            output_path=args.output_path,
        )
    )


if __name__ == "__main__":
    main()
