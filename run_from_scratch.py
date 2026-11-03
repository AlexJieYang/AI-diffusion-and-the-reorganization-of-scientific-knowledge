"""Rebuild all Python analysis from the local Parquet inputs, optionally with R."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=None)
    parser.add_argument('--with-r',action='store_true')
    args=parser.parse_args()
    project=Path(__file__).resolve().parent
    command=[sys.executable,str(project/'run_python.py')]
    if args.config:
        command.extend(['--config',str(Path(args.config).resolve())])
    if args.with_r:
        command.append('--with-r')
    subprocess.run(command,cwd=project,check=True)


if __name__=='__main__':
    main()
