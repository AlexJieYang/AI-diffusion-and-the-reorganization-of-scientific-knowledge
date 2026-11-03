"""Country portfolios over all publishing sources, using conserved paper credit."""
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import xlogy
from .runtime import dirs, csv, safe_ratio, check_memory
from .countries import country_masses, MASS
from .scope import require_scope, require_parquet


def rca_matrix(x):
    x = np.asarray(x, float)
    p = safe_ratio(x, x.sum(axis=1)[:, None])
    return safe_ratio(p, safe_ratio(x.sum(axis=0), x.sum())[None, :])


def entropy(p):
    if sparse.issparse(p):
        q = p.tocsr(copy=True)
        q.data = xlogy(q.data, q.data)
        return -np.asarray(q.sum(axis=1)).ravel()
    p = np.asarray(p, float)
    return -np.sum(xlogy(p, p), axis=-1)


def diversity_identity(x, weights=None):
    x = sparse.csr_matrix(x, dtype=float)
    totals = np.asarray(x.sum(axis=1)).ravel()
    active = totals > 0
    if weights is not None:
        weights = np.asarray(weights, float)
        active &= np.isfinite(weights) & (weights > 0)
    if not active.any():
        return dict(global_h=np.nan, within_h=np.nan, between_h=np.nan, n_countries=0,
                    effective_global=np.nan, effective_within=np.nan, effective_between=np.nan)
    x = x[active]
    totals = totals[active]
    p = sparse.diags(1/totals) @ x
    v = totals if weights is None else weights[active]
    v = v/v.sum()
    global_p = np.asarray(v @ p).ravel()
    hg = float(entropy(global_p))
    ha = float(v @ entropy(p))
    hb = hg - ha
    if hb < -1e-10:
        raise ArithmeticError('Entropy decomposition failed.')
    return dict(global_h=hg, within_h=ha, between_h=max(0, hb), n_countries=int(active.sum()),
        effective_global=float(np.exp(hg)), effective_within=float(np.exp(ha)),
        effective_between=float(np.exp(max(0, hb))))


def kitagawa(start, end):
    """Exact weighted-mean change, including entry and exit."""
    a = start[['denom', 'numer']].copy()
    b = end[['denom', 'numer']].copy()
    a = a[a.denom > 0]
    b = b[b.denom > 0]
    if a.empty or b.empty:
        return pd.DataFrame(columns=['component', 'contribution'])
    common = a.index.intersection(b.index)
    enter = b.index.difference(a.index)
    leave = a.index.difference(b.index)
    w0 = a.denom/a.denom.sum()
    w1 = b.denom/b.denom.sum()
    r0 = a.numer/a.denom
    r1 = b.numer/b.denom
    within = float(((w0.loc[common]+w1.loc[common])/2*(r1.loc[common]-r0.loc[common])).sum())
    composition = float(((r0.loc[common]+r1.loc[common])/2*(w1.loc[common]-w0.loc[common])).sum())
    entry = float((w1.loc[enter]*r1.loc[enter]).sum())
    exit = -float((w0.loc[leave]*r0.loc[leave]).sum())
    change = float(b.numer.sum()/b.denom.sum()-a.numer.sum()/a.denom.sum())
    if not np.isclose(within+composition+entry+exit, change, atol=1e-10):
        raise ArithmeticError('AI-share change accounting failed.')
    return pd.DataFrame({'component': ['within', 'composition', 'entry', 'exit', 'total'],
        'contribution': [within, composition, entry, exit, change]})


def matrices(z, key=None):
    """Sparse country-source arrays: never expand the country-source Cartesian table."""
    countries = pd.Index(sorted(z.country.unique()), name='country')
    sources = pd.Index(sorted(z.source_id.unique()), name='source_id')
    row = countries.get_indexer(z.country)
    col = sources.get_indexer(z.source_id)
    shape = (len(countries), len(sources))
    return countries.tolist(), sources.tolist(), {
        k: sparse.csr_matrix((z[k].to_numpy(float), (row, col)), shape=shape) for k in MASS}


def pairwise_distances(x, countries, minimum):
    totals = np.asarray(x.sum(axis=1)).ravel()
    active = (totals >= minimum) & (totals > 0)
    labels = np.asarray(countries)[active]
    p = (sparse.diags(1/totals[active]) @ x[active]).tocsr()
    hs = entropy(p)
    rows = [p.getrow(i) for i in range(p.shape[0])]
    for i in range(p.shape[0]):
        for j in range(i+1, p.shape[0]):
            value = float(entropy((rows[i]+rows[j])*.5)[0]-.5*(hs[i]+hs[j]))
            yield dict(country_a=labels[i], country_b=labels[j], jsd=max(0, value))


def concentration_frames(country_totals):
    if country_totals.duplicated(['year', 'country']).any():
        raise ValueError('Concentration requires one total per country-year.')
    curves, summaries = [], []
    for year, g in country_totals.groupby('year'):
        g = g[g.known > 0]
        for metric in ['known', 'ai', 'nonai', 'score']:
            p = np.sort(g[metric].to_numpy(float))
            n = len(p)
            if not n or p.sum() <= 0:
                continue
            shares = p/p.sum()
            for j, value in enumerate(np.r_[0, np.cumsum(shares)]):
                curves.append(dict(year=year, metric=metric, population_fraction=j/n,
                    cumulative_mass=value, n_countries=n))
            summaries.append(dict(year=year, metric=metric, n_countries=n,
                gini=float((2*np.arange(1, n+1)-n-1)@p/(n*p.sum())),
                hhi=float(shares@shares), top5_share=float(shares[-min(5, n):].sum()),
                top10_share=float(shares[-min(10, n):].sum())))
    return pd.DataFrame(curves), pd.DataFrame(summaries)


def add_rcas(z, time='year', baseline=None):
    """Compute margins on observed cells; observed AI zeros are retained."""
    base = z.copy()
    local = base.groupby([time, 'country'], observed=True)[MASS].transform('sum')
    world_source = base.groupby([time, 'source_id'], observed=True)[MASS].transform('sum')
    world = base.groupby(time, observed=True)[MASS].transform('sum')
    for metric, name in [('known', 'known'), ('ai', 'ai'), ('nonai', 'nonai'),
                         ('score', 'score'), ('ai_gt02', 'gt02'), ('ai_gt08', 'gt08')]:
        r = safe_ratio(safe_ratio(base[metric], local[metric]),
                       safe_ratio(world_source[metric], world[metric]))
        base['rca_'+name] = r
        base['rsca_'+name] = (r-1)/(r+1)
    base['ai_share'] = safe_ratio(base.ai, base.known)
    base['ai_lq'] = safe_ratio(base.ai_share, safe_ratio(world_source.ai, world_source.known))
    if baseline is not None:
        for metric in ['known', 'ai']:
            ref = base.source_id.map({s: v for (k, s), v in baseline.items() if k == metric})
            base['rca_fixed_'+metric] = safe_ratio(safe_ratio(base[metric], local[metric]), ref)
    return base


def build_portfolios(c):
    d = dirs(c)
    manifest = require_scope(c)
    for filename in ['country_source_masses.parquet', 'country_masses.parquet']:
        require_parquet(d['data']/filename, manifest['scope_id'])
    z = pd.read_parquet(d['data']/'country_source_masses.parquet')
    country_totals = pd.read_parquet(d['data']/'country_masses.parquet')
    base = z[z.year.between(*c['baseline_years'])].groupby('source_id')[MASS].sum()
    baseline = {(k, int(s)): float(v) for k in ['known', 'ai']
        for s, v in (base[k]/base[k].sum()).items()}
    annual = add_rcas(z, baseline=baseline)
    csv(annual, d['tables']/'country_source_rca_annual.csv')
    zz = z.copy()
    zz['period'] = (zz.year//5)*5
    per = zz.groupby(['period', 'country', 'source_id'], observed=True)[MASS].sum().reset_index()
    period = add_rcas(per, 'period', baseline)
    csv(period, d['tables']/'country_source_rca_period.csv')
    del zz, per
    diversity, pairwise = [], []
    for year, g in annual.groupby('year'):
        countries, _, x = matrices(g)
        for metric in ['known', 'ai', 'nonai', 'score']:
            for weighting in ['mass', 'equal_country']:
                v = None if weighting == 'mass' else np.ones(len(countries))
                diversity.append(dict(year=int(year), metric=metric, weighting=weighting,
                                      **diversity_identity(x[metric], v)))
            minimum = c['min_ai_annual_mass'] if metric == 'ai' else c['min_country_period_mass']
            for row in pairwise_distances(x[metric], countries, minimum):
                pairwise.append(dict(year=year, metric=metric, **row))
        check_memory(c, 'country-source portfolios')
    csv(pd.DataFrame(diversity), d['tables']/'diversity_decomposition.csv')
    csv(pd.DataFrame(pairwise, columns=['year','metric','country_a','country_b','jsd']),
        d['tables']/'country_pair_jsd.csv')
    curves, concentration = concentration_frames(country_totals)
    csv(curves, d['tables']/'country_lorenz.csv')
    csv(concentration, d['tables']/'country_concentration_summary.csv')
    # Country composition is fixed using unique-paper baseline weights.
    fixed = []
    years = range(c['fixed_support_start'], c['end_year']+1)
    common = pd.Index(sorted(z.country.unique()))
    for metric in ['known', 'ai', 'nonai']:
        totals = z.pivot_table(index='country', columns='year', values=metric, aggfunc='sum').reindex(columns=years)
        minimum = c['min_ai_annual_mass'] if metric == 'ai' else c['min_country_period_mass']
        common = common.intersection(totals.index[totals.ge(minimum).all(axis=1)])
    weights = country_totals[country_totals.year.between(*c['baseline_years'])].groupby('country').known.sum()
    common = common.intersection(weights.index[weights > 0])
    weights = weights.reindex(common)
    for year, g in z[z.country.isin(common) & z.year.ge(c['fixed_support_start'])].groupby('year'):
        countries, _, x = matrices(g)
        for metric in ['known', 'ai', 'nonai']:
            fixed.append(dict(year=year, metric=metric,
                **diversity_identity(x[metric], weights.reindex(countries).to_numpy())))
    csv(pd.DataFrame({'country':common, 'baseline_known_weight':weights/weights.sum()}),
        d['tables']/'diversity_fixed_countries.csv')
    csv(pd.DataFrame(fixed, columns=['year','metric','global_h','within_h','between_h','n_countries',
        'effective_global','effective_within','effective_between']), d['tables']/'diversity_fixed_support.csv')
    from .windowed import common_source_support, export_windowed
    support = defaultdict(list)
    for country, sid in common_source_support(annual, c):
        if (sid in base.index) and base.loc[sid, 'known'] > 0:
            support[country].append(sid)
    bknown = base.known/base.known.sum()
    country_rates = country_totals.set_index(['year','country']).ai_share
    standardized = []
    for country, g in annual.groupby('country'):
        ss = sorted(support[country])
        if not ss:
            continue
        w = bknown.reindex(ss)
        retained = float(w.sum())
        w = w/retained
        for year, gg in g[g.source_id.isin(ss)].groupby('year'):
            rates = gg.set_index('source_id').ai_share.reindex(ss)
            if rates.notna().all():
                standardized.append(dict(country=country, year=year,
                    standardized_share=float(rates@w), raw_common_share=float(gg.ai.sum()/gg.known.sum()),
                    retained_baseline_source_mass=retained, n_sources=len(ss),
                    retained_country_scored_mass=float(gg.known.sum()),
                    country_all_scored_share=country_rates.get((year,country), np.nan)))
    cols = ['country','year','standardized_share','raw_common_share','retained_baseline_source_mass',
            'n_sources','retained_country_scored_mass','country_all_scored_share']
    csv(pd.DataFrame(standardized, columns=cols), d['tables']/'ai_standardized_common_support.csv')
    decomp = []
    for denom, numer, label in [('known','ai','binary_share'), ('known','score','mean_score')]:
        a = z[z.year.between(*c['baseline_years'])].groupby(['country','source_id'])[[denom,numer]].sum().rename(columns={denom:'denom',numer:'numer'})
        b = z[z.year.between(*c['comparison_years'])].groupby(['country','source_id'])[[denom,numer]].sum().rename(columns={denom:'denom',numer:'numer'})
        q = kitagawa(a, b)
        if len(q):
            q['measure'] = label
            q['partition'] = 'country x publishing source; country-attributed paper credit'
            decomp.append(q)
    csv(pd.concat(decomp, ignore_index=True) if decomp else pd.DataFrame(
        columns=['component','contribution','measure','partition']), d['tables']/'ai_change_accounting.csv')
    period['state'] = 'Unobserved / low support'
    supported = (period.known.ge(c['min_cell_period_mass'])
        & period.groupby(['period','country']).known.transform('sum').ge(c['min_country_period_mass'])
        & period.groupby(['period','country']).ai.transform('sum').ge(c['min_ai_period_mass'])
        & period.rca_ai.notna() & period.rca_nonai.notna())
    aa = period.rca_ai.ge(1)
    nn = period.rca_nonai.ge(1)
    period.loc[supported, 'state'] = np.select(
        [aa[supported]&nn[supported], aa[supported]&~nn[supported], ~aa[supported]&nn[supported]],
        ['Both specialized','AI-positive specialized','Other-scored specialized'], default='Neither specialized')
    csv(period[['period','country','source_id','known','state']], d['tables']/'specialization_states.csv')
    export_windowed(annual, c, country_totals)
    from .publication_tables import export_publication_tables
    export_publication_tables(c)
    return annual
