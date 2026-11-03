"""Pooled country-source portfolios and common-source standardization support."""
import numpy as np
import pandas as pd
from .runtime import csv, dirs


def windowed_portfolios(annual, c, country_totals=None):
    from .portfolios import matrices, diversity_identity, pairwise_distances
    from .countries import MASS
    width = int(c.get('portfolio_window_years', 5))
    if width < 1:
        raise ValueError('portfolio_window_years must be positive.')
    z = annual[annual.year.between(c['trend_start'], c['end_year'])].copy()
    z['year'] = (z.year//width)*width
    z = z.groupby(['year','country','source_id'], observed=True)[MASS].sum().reset_index()
    diversity, pairs, fixed, support = [], [], [], []
    for year, g in z.groupby('year'):
        countries, _, x = matrices(g)
        info = dict(year=int(year), period_end=min(int(year)+width-1, c['end_year']), window_years=width)
        for metric in ['known','ai','nonai','score']:
            for weighting in ['mass','equal_country']:
                weights = None if weighting == 'mass' else np.ones(len(countries))
                diversity.append(dict(**info, metric=metric, weighting=weighting,
                    **diversity_identity(x[metric], weights)))
        total = np.asarray(x['ai'].sum(axis=1)).ravel()
        active = (total >= c['min_ai_period_mass']) & (total > 0)
        n = int(active.sum())
        support.append(dict(**info, n_ai_countries=n, n_pairs=n*(n-1)//2,
            ai_fractional_credit=float(total.sum()), minimum_ai_credit=c['min_ai_period_mass']))
        for row in pairwise_distances(x['ai'], countries, c['min_ai_period_mass']):
            pairs.append(dict(**info, metric='ai', **row))
    years = list(range(c['fixed_support_start'], c['end_year']+1, width))
    if c['fixed_support_start'] % width:
        raise ValueError('fixed_support_start must align with portfolio windows.')
    common = pd.Index(sorted(z.country.unique()), name='country')
    for metric in ['known','ai','nonai']:
        totals = z.pivot_table(index='country', columns='year', values=metric, aggfunc='sum').reindex(columns=years)
        minimum = c['min_ai_period_mass'] if metric == 'ai' else c['min_country_period_mass']
        common = common.intersection(totals.index[totals.ge(minimum).all(axis=1)])
    weight_source = country_totals if country_totals is not None else annual
    baseline = weight_source[weight_source.year.between(*c['baseline_years'])].groupby('country').known.sum()
    weights = baseline.reindex(common, fill_value=0)
    common = common.intersection(weights.index[weights > 0])
    weights = weights.reindex(common)
    for year, g in z[z.country.isin(common)&z.year.isin(years)].groupby('year'):
        countries, _, x = matrices(g)
        for metric in ['known','ai','nonai']:
            fixed.append(dict(year=int(year), period_end=min(int(year)+width-1,c['end_year']),
                window_years=width, metric=metric,
                **diversity_identity(x[metric], weights.reindex(countries).to_numpy())))
    return dict(
        diversity_period_decomposition=pd.DataFrame(diversity),
        country_pair_jsd_period=pd.DataFrame(pairs, columns=['year','period_end','window_years','metric','country_a','country_b','jsd']),
        diversity_period_support=pd.DataFrame(support),
        diversity_fixed_support_period=pd.DataFrame(fixed, columns=['year','period_end','window_years','metric',
            'global_h','within_h','between_h','n_countries','effective_global','effective_within','effective_between']),
        diversity_fixed_countries_period=pd.DataFrame({'country':common,'baseline_known_weight':weights/weights.sum()}))


def export_windowed(annual, c, country_totals=None):
    frames = windowed_portfolios(annual, c, country_totals)
    for name, frame in frames.items():
        csv(frame, dirs(c)['tables']/(name+'.csv'))
    return frames


def common_source_support(annual, c):
    """Require scored credit in every declared window and an observed target-year rate."""
    width = int(c.get('standardization_window_years', 5))
    if width < 1 or c['fixed_support_start'] % width:
        raise ValueError('Standardization windows must be positive and aligned.')
    z = annual[annual.year.between(c['fixed_support_start'], c['end_year'])].copy()
    z['window'] = (z.year//width)*width
    totals = z.groupby(['country','source_id','window'], observed=True).known.sum().unstack('window')
    totals = totals.reindex(columns=range(c['fixed_support_start'],c['end_year']+1,width))
    good = totals.ge(c['min_cell_period_mass']).all(axis=1)
    target = annual[annual.year.eq(c['end_year'])].groupby(['country','source_id']).known.sum()
    good &= target.reindex(totals.index,fill_value=0).gt(0)
    return set(totals.index[good])
