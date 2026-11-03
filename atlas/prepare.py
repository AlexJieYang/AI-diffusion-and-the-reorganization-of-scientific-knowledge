import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .runtime import (source, dirs, names, batches, exact_ids, write_parquet,
                      csv, json_write, check_memory, require)
from .sources import source_ids, build_source_catalog

BASE_NUMERIC = ['n_references', 'mean_ref_age', 'team_size', 'n_countries',
    'is_international', 'mean_geo_km', 'c5', 'c5_norm', 'c5_pct', 'top10_credit', 'mean_prior_papers']
METRICS = {
    'idr': ['n_field_refs', 'ref_field_coverage', 'ref_field_variety', 'ref_field_shannon',
            'ref_field_simpson', 'ref_field_rao', 'ref_outside_primary_share'],
    'semantic': ['emb_ref_coverage', 'emb_ref_pair_mean_distance',
                 'emb_ref_mean_distance', 'emb_ref_centroid_distance'],
    'history': ['emb_field_history_count', 'emb_field_history_distance']}


def validate_inputs(c):
    """Validate required inputs before clearing any previous analysis output."""
    keys = ['papers', 'ai', 'members', 'idr', 'semantic', 'history']
    if c.get('edges_enabled', True):
        keys.append('edges')
    require(c, keys)
    schemas = {'papers': ['paper_id', 'year', 'source_id', 'is_conference', 'c5_norm'],
        'ai': ['paper_id', 'AIProb'], 'members': ['paper_id', 'year', 'entity_id']}
    schemas.update({key: ['paper_id']+columns for key, columns in METRICS.items()})
    for key, columns in schemas.items():
        missing = set(columns) - set(names(source(c, key)))
        if missing:
            raise KeyError(f'{source(c, key)}: missing {sorted(missing)}')


def preflight(c):
    validate_inputs(c)
    d = dirs(c)
    rows = []
    for key in c['paths']:
        if not c['paths'][key] or (key == 'edges' and not c['edges_enabled']):
            continue
        path = source(c, key)
        pf = pq.ParquetFile(path)
        rows.append(dict(key=key, path=str(path), rows=pf.metadata.num_rows,
            columns=';'.join(pf.schema_arrow.names), size_gib=path.stat().st_size/2**30))
    z = pd.DataFrame(rows)
    csv(z, d['tables']/'input_catalog.csv')
    json_write(c, d['status']/'settings.json')
    _, cap = check_memory(c)
    print(f'RSS limit {cap:.1f} GiB; use a fresh kernel for each notebook.')
    print(z[['key', 'rows', 'size_gib']].to_string(index=False))
    return z


def attach(master, index, c, path, cols, required=True):
    arrays = {x: np.full(len(master), np.nan, dtype='float64') for x in cols}
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        for x in cols:
            master[x] = arrays[x]
        return
    wanted = [x for x in cols if x in names(path)]
    if required and len(wanted) != len(cols):
        raise KeyError(f'{path}: missing {set(cols)-set(wanted)}')
    seen = np.zeros(len(master), dtype=bool)
    for b in batches(path, ['paper_id']+wanted, c['batch_rows']):
        ids = exact_ids(b.paper_id)
        if pd.Index(ids).has_duplicates:
            raise ValueError(f'Duplicate paper_id in {path}')
        ix = index.get_indexer(ids)
        ok = ix >= 0
        ix = ix[ok]
        if seen[ix].any():
            raise ValueError(f'Duplicate paper_id across row groups in {path}')
        seen[ix] = True
        for x in wanted:
            arrays[x][ix] = pd.to_numeric(b.loc[ok, x], errors='coerce').to_numpy(
                dtype='float64', na_value=np.nan)
        check_memory(c, f'attach {path.name}')
    print(f'{path.name}: {int(seen.sum()):,} matched core records', flush=True)
    for x in cols:
        master[x] = arrays[x]


def build_master(c):
    from .scope import record_scope, assert_sample
    validate_inputs(c)
    d = dirs(c)
    parts = []
    for b in batches(source(c, 'ai'), ['paper_id', 'AIProb'], c['batch_rows']):
        ids = exact_ids(b.paper_id)
        prob = pd.to_numeric(b.AIProb, errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
        if (np.isinf(prob) | (np.isfinite(prob) & ((prob < 0) | (prob > 1)))).any():
            raise ValueError('AIProb must be missing or finite in [0,1].')
        parts.append(pd.DataFrame({'paper_id': ids, 'AIProb': prob}))
    if not parts:
        raise ValueError('No AI score input records.')
    scores = pd.concat(parts, ignore_index=True)
    del parts
    if scores.paper_id.duplicated().any():
        raise ValueError('Duplicate paper_id in AI input.')
    raw_ai_records = len(scores)
    scores = scores.dropna(subset=['AIProb']).set_index('paper_id')
    if scores.empty:
        raise ValueError('No observed AI scores.')
    score_index = scores.index
    score_values = scores.AIProb.to_numpy(dtype='float64')
    positive_flow = [dict(stage='Observed scores in AI input', n_scored_papers=len(scores),
        n_ai_positive=int(scores.AIProb.gt(.5).sum()))]
    keep = ['paper_id', 'year', 'source_id', 'is_conference']+BASE_NUMERIC
    available = names(source(c, 'papers'))
    flow = dict(input_core_records=0, excluded_no_ai_score=0, excluded_invalid_or_outside_year=0)
    matched_n = matched_positive = 0
    frames = []
    for b in batches(source(c, 'papers'), [x for x in keep if x in available], c['batch_rows']):
        flow['input_core_records'] += len(b)
        ids = exact_ids(b.paper_id)
        pos = score_index.get_indexer(ids)
        scored = pos >= 0
        year = pd.to_numeric(b.year, errors='coerce')
        good_year = year.between(c['ai_start'], c['end_year']) & year.mod(1).eq(0)
        flow['excluded_no_ai_score'] += int((~scored).sum())
        flow['excluded_invalid_or_outside_year'] += int((scored & ~good_year).sum())
        matched_n += int(scored.sum())
        matched_positive += int((score_values[pos[scored]] > .5).sum())
        valid = scored & good_year
        if not valid.any():
            continue
        z = b.loc[valid].copy()
        z['paper_id'] = ids[valid]
        z['year'] = year.loc[valid].astype('int16')
        z['source_id'] = source_ids(z.source_id)
        z['AIProb'] = score_values[pos[valid]]
        z['is_conference'] = pd.to_numeric(z.is_conference, errors='coerce').astype('float32')
        for x in BASE_NUMERIC:
            z[x] = pd.to_numeric(z[x], errors='coerce').astype('float64') if x in z else np.nan
        frames.append(z)
        check_memory(c, 'read scored core papers')
    if not frames:
        raise ValueError('No scored papers in the publication window.')
    m = pd.concat(frames, ignore_index=True)
    del frames, scores, score_index, score_values
    gc.collect()
    if m.paper_id.duplicated().any():
        raise ValueError('Duplicate eligible paper_id in core input.')
    assert_sample(m, c, 'master')
    flow['retained_records'] = len(m)
    manifest = record_scope(c, m, flow)
    catalog = build_source_catalog(m, c)
    csv(catalog, d['tables']/'source_catalog.csv')
    csv(pd.DataFrame([dict(step=k, n=v) for k, v in flow.items()]), d['tables']/'sample_flow.csv')
    positive_flow.extend([
        dict(stage='Matched to core before year restriction', n_scored_papers=matched_n, n_ai_positive=matched_positive),
        dict(stage='Retained research population', n_scored_papers=len(m), n_ai_positive=int(m.AIProb.gt(.5).sum()))])
    pf = pd.DataFrame(positive_flow)
    pf['n_other_scored'] = pf.n_scored_papers - pf.n_ai_positive
    csv(pf, d['tables']/'ai_positive_flow.csv')
    index = pd.Index(m.paper_id)
    optional = source(c, 'papers').parents[1]/'metrics/team_affiliation_diversity.parquet'
    attach(m, index, c, optional, ['country_author_coverage', 'team_country_simpson'], False)
    for key, columns in METRICS.items():
        attach(m, index, c, source(c, key), columns)
    write_parquet(m, d['data']/'papers.parquet', c['batch_rows'])
    summary = flow | dict(known_ai=len(m), ai_gt_05=int(m.AIProb.gt(.5).sum()),
        n_sources=int(m.source_id.nunique()), n_example_sources=int(catalog.selected.sum()),
        raw_ai_input_records=raw_ai_records)
    json_write(summary, d['status']/'population.json')
    print(summary, flush=True)
    return summary
