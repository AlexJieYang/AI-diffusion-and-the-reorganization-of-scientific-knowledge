from bootstrap import configure
configure()
import argparse
from atlas.runtime import config,run_stage,fresh_outputs,dirs
from atlas.prepare import preflight,build_master,validate_inputs
from atlas.describe import build_descriptions
from atlas.portfolios import country_masses,build_portfolios
from atlas.knowledge import profile_comparisons,reference_cohorts

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=None)
    parser.add_argument('--resume',action='store_true',help='Explicitly reuse the current scoped output folder for selected stages.')
    parser.add_argument('--with-r',action='store_true',help='Run the R models with Rscript before rendering figures.')
    parser.add_argument('--steps',nargs='+',default=['preflight','master','descriptions','countries','portfolios','profiles','cohorts','figures'])
    args=parser.parse_args();c=config(args.config)
    if args.with_r:
        import shutil
        if not shutil.which('Rscript'):
            raise FileNotFoundError('Rscript was not found. Configure R first, or run the Python command without --with-r and use notebook 05 separately.')
    if not args.resume:
        if 'master' not in args.steps:raise ValueError('A partial run needs --resume. The default command starts at master.')
        validate_inputs(c)
        fresh_outputs(c)
    from atlas.figures import make_figures
    stages=dict(preflight=preflight,master=build_master,descriptions=build_descriptions,countries=country_masses,
                portfolios=build_portfolios,profiles=profile_comparisons,cohorts=reference_cohorts,figures=make_figures)
    for step in args.steps:
        if step not in stages:raise ValueError(f'Unknown step {step}')
        if step=='figures' and args.with_r:
            import subprocess,shutil
            from pathlib import Path
            rscript=shutil.which('Rscript')
            if not rscript:raise FileNotFoundError('Rscript was not found. Run 05_R_Conditional_Associations.ipynb using its R kernel.')
            # R reads the exact same config, even for a CLI --config override.
            project=Path(__file__).resolve().parent
            subprocess.run([rscript,str(project/'R'/'run.R'),str(project),str(Path(args.config).resolve()) if args.config else str(project/'config.json')],check=True)
        run_stage(c,step,stages[step])

if __name__=='__main__':main()
