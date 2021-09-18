import subprocess


def download_official():
    """ Download official dataset from aws s3 servers. """

    # Dict from s3 source path to target location
    source_to_target = {
        # Predict Modality dataset
        's3://openproblems-bio/public/phase1-data/predict_modality/':
            'data/official/predict_modality/',
        
        # Explore dataset
        's3://openproblems-bio/public/explore/':
            'data/official/explore/',
    }

    for source_path, target_path in source_to_target.items():
        command = r'aws s3 sync --no-sign-request ' + source_path + ' ' + target_path
        subprocess.call(command, shell=True)


download_official()