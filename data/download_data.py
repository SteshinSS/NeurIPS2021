"""
The script for downloading datasets. Need AWS CLI to work.
Run it from the repo's root directory: 
    python data/download_data.py

At the moment it downloads every official dataset.
"""

import argparse
import subprocess


def download_official():
    """Download official dataset from aws s3 servers."""

    # Dict from s3 source path to target location
    source_to_target = {
        # Predict Modality
        "s3://openproblems-bio/public/phase1-data/predict_modality/": "data/official/predict_modality/",
        # Match Modality
        "s3://openproblems-bio/public/phase1-data/match_modality/": "data/official/match_modality/",
        # Joint Embedding
        "s3://openproblems-bio/public/phase1-data/joint_embedding/": "data/official/joint_embedding/",
        # Common
        "s3://openproblems-bio/public/phase1-data/common/": "data/official/common/",
        # Explore
        "s3://openproblems-bio/public/explore/": "data/official/explore/",
    }

    for source_path, target_path in source_to_target.items():
        print(f"Download {source_path} to {target_path}...")
        command = r"aws s3 sync --no-sign-request " + source_path + " " + target_path
        subprocess.call(command, shell=True)
        print()


def download_our(source_to_target: dict):
    """Download datasets from our aws.

    Args:
        source_to_target (dict): dict from s3 paths to local paths.
    """
    for source_path, target_path in source_to_target.items():
        print(f"Download {source_path} to {target_path}...")
        command = r"aws s3 sync " + source_path + " " + target_path
        subprocess.call(command, shell=True)
        print()


def download_preprocessed():
    """Download preprocessed datasets from our aws.
    Use it if you train models.
    """
    source_to_target = {"s3://nips2021/data/preprocessed": "data/preprocessed"}
    download_our(source_to_target)


def download_raw():
    """Download raw datasets from our aws.
    Use it if you preprocess data.
    """
    source_to_target = {"s3://nips2021/data/raw": "data/raw"}
    download_our(source_to_target)


def get_parser():
    parser = argparse.ArgumentParser(description="Dataset downloader")
    parser.add_argument(
        "--all", action="store_true", default=False, help="Download all datasets"
    )
    parser.add_argument(
        "--official",
        action="store_true",
        default=False,
        help="Download the official dataset",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        default=False,
        help="Download our preprocessed datasets",
    )
    parser.add_argument(
        "--raw", action="store_true", default=False, help="Download our raw datasets"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    is_official = args.official
    is_preprocessed = args.preprocessed
    is_raw = args.raw
    if args.all:
        is_official = True
        is_preprocessed = True
        is_raw = True

    if is_official:
        download_official()
    if is_preprocessed:
        download_preprocessed()
    if is_raw:
        download_raw()
    if not is_official and not is_preprocessed and not is_raw:
        print('Run with --help to see help.')
