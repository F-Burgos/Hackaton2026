from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project.src.evaluation.contrastive_report import (  # noqa: E402
    ContrastiveReportConfig,
    write_contrastive_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Markdown report for a contrastive run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    write_contrastive_report(
        ContrastiveReportConfig(
            run_dir=args.run_dir,
            export_dir=args.export_dir,
            output_path=args.output_path,
        )
    )


if __name__ == "__main__":
    main()
