from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.evaluation.embedding_diagnostics import (  # noqa: E402
    EmbeddingDiagnosticsConfig,
    run_embedding_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-downstream diagnostics on embeddings.")
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_embedding_diagnostics(
        EmbeddingDiagnosticsConfig(
            embeddings_path=args.embeddings_path,
            output_dir=args.output_dir,
            n_neighbors=args.n_neighbors,
            make_plot=not args.no_plot,
        )
    )


if __name__ == "__main__":
    main()
