"""Run SVT experiments on recommender matrices."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SVT reconstruction on recommender data."
    )
    parser.add_argument(
        "--input", type=str, required=False, help="Path to ratings matrix."
    )
    parser.add_argument(
        "--missing-rate", type=float, default=0.8, help="Fraction of missing entries."
    )
    parser.parse_args()

    print("SVT recommender experiment scaffold ready.")


if __name__ == "__main__":
    main()
