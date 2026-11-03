"""Run this with the desired Python environment, not an unrelated pip."""
import sys, subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.check_call([sys.executable,'-m','pip','install','-r',str(root/'requirements.txt')])
subprocess.check_call([sys.executable,'-m','ipykernel','install','--user','--name','joi-ai',
                       '--display-name','Python - JOI'])
print('Kernel installed for',sys.executable)
