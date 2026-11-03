"""Refresh support rules and figures using existing local analysis outputs."""
from bootstrap import configure
configure()
import argparse
from atlas.runtime import config,run_stage
from atlas.scope import require_scope
from atlas.portfolios import build_portfolios
from atlas.knowledge import profile_comparisons
from atlas.describe import refresh_surface_support
from atlas.figures import make_figures


def refresh(c):
    require_scope(c)
    for name,fn in [('portfolios',build_portfolios),('profiles',profile_comparisons),
                    ('surface_support',refresh_surface_support)]:
        run_stage(c,name,fn)
    return run_stage(c,'figures',make_figures)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=None)
    args=parser.parse_args()
    refresh(config(args.config))
