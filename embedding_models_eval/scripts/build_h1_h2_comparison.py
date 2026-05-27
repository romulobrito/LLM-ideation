"""
CLI wrapper to consolidate H1 and H2 outputs.
"""

from __future__ import annotations

import argparse

from embedding_models_eval.h1_h2_comparison import consolidate_h1_h2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build H1+H2 consolidated comparison files")
    parser.add_argument(
        "--h1-path",
        required=True,
        help="H1 macro CSV file or directory containing comparacao_modelos_macro.csv",
    )
    parser.add_argument(
        "--h2-path",
        required=True,
        help="H2 macro CSV file or directory containing comparacao_modelos_macro.csv",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for consolidated H1+H2 files",
    )
    args = parser.parse_args()

    outputs = consolidate_h1_h2(
        h1_path=args.h1_path,
        h2_path=args.h2_path,
        out_dir=args.out_dir,
    )
    print("Consolidation completed.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
