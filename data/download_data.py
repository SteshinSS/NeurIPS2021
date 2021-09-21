"""
The script for downloading datasets. Need AWS CLI to work.
Run it from the repo's root directory: 
    python data/download_data.py

At the moment it downloads every official dataset.
"""

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
        command = r"aws s3 sync --no-sign-request " + source_path + " " + target_path
        subprocess.call(command, shell=True)


download_official()
