#!/usr/bin/env python
import os
import pathlib
import argparse
import numpy as np
from scope.utils import parse_load_config


BASE_DIR = pathlib.Path.cwd()
BASE_DIR_PREDS = BASE_DIR

config = parse_load_config()

path_to_preds = config['inference']['path_to_preds']
if path_to_preds is not None:
    BASE_DIR_PREDS = pathlib.Path(path_to_preds)


def get_parser():
    """
    Parse the options given on the command-line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dirname",
        type=str,
        default='inference',
        help="Directory name for inference slurm scripts",
    )
    parser.add_argument(
        "-f", "--filetype", default="slurm", help="Type of job submission file"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="bhealy",
        help="HPC username",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default='dnn',
        help="dnn or xgb",
    )

    return parser


def filter_completed(fields, algorithm):

    fields_copy = fields.copy()

    for field in fields:
        searchDir = BASE_DIR_PREDS / f'preds_{algorithm}' / f'field_{field}'
        searchDir.mkdir(parents=True, exist_ok=True)
        generator = searchDir.iterdir()
        has_parquet = np.sum([x.suffix == '.parquet' for x in generator]) > 0

        if has_parquet:
            fields_copy.remove(field)

    print('Models remaining to run: ', len(fields_copy))

    return fields_copy


def run_job(field, subfile, alg):
    sbatchstr = f"sbatch --export=FID={field} --job-name='field_{field}_{alg}_inference'  {subfile}"
    print(sbatchstr)
    os.system(sbatchstr)


def main():
    # Parse command line
    parser = get_parser()
    args, _ = parser.parse_known_args()

    filetype = args.filetype
    print(f'filetype: {filetype}')
    dirname = args.dirname
    print(f'dirname: {dirname}')

    slurmDir = str(BASE_DIR / dirname)

    print(f'slurmDir: {slurmDir}')

    fields = config['inference'].get('fields_to_run')
    fields_to_exclude = config.get('fields_to_exclude')

    if (fields is not None) and (fields_to_exclude is not None):
        # Drop fields_to_exclude from fields
        fields = list(set(fields).difference(fields_to_exclude))

    algorithm = args.algorithm

    subDir = os.path.join(slurmDir, filetype)
    print(f'subDir: {subDir}')
    subfile = os.path.join(subDir, '%s.sub' % filetype)
    print(f'subfile: {subfile}')
    fields_remaining = filter_completed(
        fields,
        algorithm,
    )

    for field in fields_remaining:
        # Only run jobs from tags_remaining_to_run list
        run_job(field, subfile, algorithm)

    os.system(f"squeue -u {args.user}")

    print('The final jobs in the run have been queued - breaking loop.')
    print('Run "squeue -u <username>" to check status of remaining jobs.')
