"""Conserved country credit, partitioned by each paper's publishing source."""
import gc

import numpy as np
import pandas as pd

from .runtime import (dirs, source, batches, exact_ids, SingleWriter, analytical_db,
                      sql_string, csv, json_write, write_parquet, safe_ratio, check_memory,query_batches)
from .scope import require_scope, assert_sample

MASS = ['known','ai','nonai','score','ai_gt02','ai_gt08']
PROFILES = ['ref_field_simpson','emb_ref_pair_mean_distance','mean_ref_age','is_international']


def country_masses(c):
    d = dirs(c)
    require_scope(c)
    m = pd.read_parquet(d['data']/'papers.parquet', columns=['paper_id','year','source_id','AIProb']+PROFILES)
    assert_sample(m,c,'country input')
    index = pd.Index(m.paper_id)
    candidate = d['data']/'country_members_candidate.parquet'
    with SingleWriter(candidate) as writer:
        for b in batches(source(c,'members'), ['paper_id','year','entity_id'], c['batch_rows']):
            ids = exact_ids(b.paper_id)
            positions = index.get_indexer(ids)
            keep = positions>=0
            if not keep.any():
                continue
            selected = positions[keep]
            b = b.loc[keep].copy()
            country = b.entity_id.astype('string').str.strip().str.upper()
            if not country.str.fullmatch('[A-Z]{2}',na=False).all():
                raise ValueError('Selected country memberships need nonempty two-letter country codes.')
            years = pd.to_numeric(b.year,errors='raise').to_numpy()
            if not np.array_equal(years,m.year.to_numpy()[selected]):
                raise ValueError('Selected country member year disagrees with the paper year.')
            writer.write(pd.DataFrame({'paper_id':ids[keep], 'country':country.to_numpy()}))
        if writer.writer is None:
            writer.write(pd.DataFrame({'paper_id':pd.Series(dtype='uint64'),
                                       'country':pd.Series(dtype='string')}))
    con = analytical_db(c)
    try:
        query = f'''WITH unique_members AS (
              SELECT DISTINCT paper_id,country FROM read_parquet({sql_string(candidate)}))
            SELECT paper_id,country,1.0/COUNT(*) OVER(PARTITION BY paper_id) AS weight
            FROM unique_members ORDER BY paper_id,country'''
        with SingleWriter(d['data']/'country_members_scoped.parquet') as writer:
            reader = query_batches(con,query,c['batch_rows'])
            schema = reader.schema
            for b in reader:
                writer.write(b.to_pandas())
            if writer.writer is None:
                import pyarrow as pa
                writer.write(pa.Table.from_batches([],schema=schema).to_pandas())
    finally:
        con.close()
        candidate.unlink(missing_ok=True)
    from .describe import merge_sums
    credit = np.zeros(len(m),float)
    parts = []
    set_parts = []
    member_rows = 0
    for b in batches(d['data']/'country_members_scoped.parquet',size=c['batch_rows']):
        at = index.get_indexer(b.paper_id)
        w = b.weight.to_numpy(float)
        np.add.at(credit,at,w)
        p = m.AIProb.to_numpy()[at]
        ai = p>.5
        z = pd.DataFrame(dict(year=m.year.to_numpy()[at],source_id=m.source_id.to_numpy()[at],
            country=b.country.to_numpy(),known=w,ai=w*ai,nonai=w*~ai,
            score=w*p,ai_gt02=w*(p>.2),ai_gt08=w*(p>.8)))
        for name in PROFILES:
            val = m[name].to_numpy()[at]
            valid = np.isfinite(val)
            z[name+'_denom'] = w*valid
            z[name+'_numer'] = w*np.where(valid,val,0)
        values=MASS+[name+suffix for name in PROFILES for suffix in ['_denom','_numer']]
        parts.append(z.groupby(['year','country'],observed=True)[values].sum().reset_index())
        set_parts.append(z.groupby(['year','country','source_id'],observed=True)[MASS].sum().reset_index())
        if len(parts)>=10:
            parts=[merge_sums(parts,['year','country'])]
            set_parts=[merge_sums(set_parts,['year','country','source_id'])]
        member_rows += len(b)
        check_memory(c,'country aggregation')
    if not parts:
        raise ValueError('No selected scored paper has country information.')
    if not np.isclose(credit[credit>0],1,atol=1e-10).all():
        raise ArithmeticError('Country credit must sum to one per attributable paper.')
    z = merge_sums(parts,['year','country'])
    z['ai_share'] = safe_ratio(z.ai,z.known)
    z['mean_aiprob'] = safe_ratio(z.score,z.known)
    for name in PROFILES:
        z[name+'_mean'] = safe_ratio(z[name+'_numer'],z[name+'_denom'])
    write_parquet(z,d['data']/'country_masses.parquet')
    venues=merge_sums(set_parts,['year','country','source_id'])
    venues['ai_share']=safe_ratio(venues.ai,venues.known)
    venues['mean_aiprob']=safe_ratio(venues.score,venues.known)
    write_parquet(venues,d['data']/'country_source_masses.parquet')
    coverage = pd.DataFrame(dict(year=m.year,papers=1,country_known=credit>0,
        ai_positive=m.AIProb.gt(.5).astype('int64'),
        country_known_ai_positive=((credit>0)&m.AIProb.gt(.5)).astype('int64')))
    coverage = coverage.groupby('year').sum().reset_index()
    csv(coverage,d['tables']/'country_attribution_coverage.csv')
    json_write(dict(country_member_rows=member_rows,
        scored_papers=len(m),country_attributed_papers=int((credit>0).sum()),
        country_credit_rule='One total unit per unique paper, divided equally across its distinct countries.',
        source_credit_rule='Each paper has one publishing source; source totals conserve country-paper credit.'),
        d['status']/'country_scope.json')
    del m,index,credit
    gc.collect()
    return venues


