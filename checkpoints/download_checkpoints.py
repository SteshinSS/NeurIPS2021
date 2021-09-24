import subprocess

source_path = "s3://nips2021/checkpoints/"
target_path = "checkpoints"
command = r"aws s3 sync " + source_path + " " + target_path
