import hashlib
import numpy as np
import pandas as pd
from .runtime import dirs,source,names,batches,exact_ids,csv,safe_ratio,check_memory,json_write
from .describe import PROFILE_MEASURES
from .scope import require_scope,read_table,require_parquet


def profile_comparisons(c):
    """Within-source comparisons of the supplied paper-level indicators."""
    d=dirs(c)
    s=read_table(c,'source_year_groups.csv')
    s=s[s.year.between(c['trend_start'],c['end_year']) & s.ai_group.isin([0,1])]
    years=list(range(c['trend_start'],c['end_year']+1))
    catalog=read_table(c,'source_catalog.csv').set_index('source_id')
    contrasts=[];standard=[];support_rows=[]
    keys=['year','source_id']
    for x in PROFILE_MEASURES:
        cols=keys+[x+'_n',x+'_mean',x+'_coverage']
        a=s.loc[s.ai_group.eq(0),cols].rename(columns={x+'_n':'n0',x+'_mean':'low_score_mean',x+'_coverage':'coverage0'})
        b=s.loc[s.ai_group.eq(1),cols].rename(columns={x+'_n':'n1',x+'_mean':'high_score_mean',x+'_coverage':'coverage1'})
        detail=a.merge(b,on=keys,how='outer',validate='one_to_one')
        detail[['n0','n1']]=detail[['n0','n1']].fillna(0).astype('int64')
        detail['supported']=(detail.n0.ge(c['min_profile_cell_n']) & detail.n1.ge(c['min_profile_cell_n'])
                             & detail.low_score_mean.notna() & detail.high_score_mean.notna())
        detail['metric']=x
        detail['source_label']=detail.source_id.map(catalog.source_label)
        support_rows.append(detail[keys+['metric','source_label','n0','n1','supported']])
        t=detail[detail.supported].copy()
        t['difference']=t.high_score_mean-t.low_score_mean
        contrasts.append(t)
        previous=None;segment=0
        for year in years:
            all_measurable=detail[detail.year.eq(year)]
            g=t[t.year.eq(year)]
            ids=sorted(int(v) for v in g.source_id)
            signature=hashlib.sha256(';'.join(map(str,ids)).encode()).hexdigest()
            changed=previous is not None and signature!=previous
            if changed:segment+=1
            row=dict(year=year,metric=x,n0=int(g.n0.sum()),n1=int(g.n1.sum()),n_sources=len(ids),
                retained_sources='; '.join(catalog.loc[ids,'source_label'].astype(str)),
                retained_source_ids=';'.join(map(str,ids)),support_key=signature,
                support_changed=changed,support_segment=segment,
                comparison_set=f'{len(ids)} common sources' if ids else 'No supported sources',
                other_scored_retained_fraction=float(safe_ratio(g.n0.sum(),all_measurable.n0.sum())),
                ai_positive_retained_fraction=float(safe_ratio(g.n1.sum(),all_measurable.n1.sum())),
                source_adjusted_difference=np.nan,raw_difference=np.nan,
                within_component=np.nan,composition_component=np.nan)
            if len(g):
                n0=g.n0.to_numpy();n1=g.n1.to_numpy()
                m0=g.low_score_mean.to_numpy();m1=g.high_score_mean.to_numpy()
                w0=n0/n0.sum();w1=n1/n1.sum();pool=(n0+n1)/(n0+n1).sum()
                within=float(np.sum((w0+w1)/2*(m1-m0)))
                composition=float(np.sum((m0+m1)/2*(w1-w0)))
                raw=float(np.sum(w1*m1)-np.sum(w0*m0))
                if not np.isclose(within+composition,raw,atol=1e-8):
                    raise ArithmeticError('Source-composition accounting failed.')
                row.update(source_adjusted_difference=float(np.sum(pool*(m1-m0))),
                    raw_difference=raw,within_component=within,composition_component=composition)
            standard.append(row);previous=signature
    result=pd.DataFrame(standard)
    csv(pd.concat(contrasts,ignore_index=True),d['tables']/'profile_source_year_contrasts.csv')
    csv(pd.concat(support_rows,ignore_index=True),d['tables']/'profile_source_year_support.csv')
    csv(result,d['tables']/'profile_source_adjusted_contrasts.csv')
    from .publication_tables import export_profile_support_tables
    export_profile_support_tables(c,result)
    return result


def reference_cohorts(c):
    # Plot-level aggregation of existing edges, never recalculation of paper metrics.
    from .runtime import analytical_db,query_batches,sql_string,sql_identifier,SingleWriter
    from .scope import assert_sample
    d=dirs(c);require_scope(c)
    columns=['citing_year','cited_year','ai_group','reference_ai_group','edges','reference_age']
    if not c['edges_enabled']:
        result=pd.DataFrame(columns=columns)
        csv(result,d['tables']/'reference_cohorts.csv')
        return result
    path=source(c,'edges')
    if not path.is_file():raise FileNotFoundError(path)
    available=names(path);col=c['edge_columns']
    if not {col['citing'],col['cited']}<=set(available):
        candidates=[('citing_id','cited_id'),('citing_paper_id','cited_paper_id'),('citing_paperid','cited_paperid')]
        match=[p for p in candidates if set(p)<=set(available)]
        if len(match)!=1:raise KeyError('Configure edge_columns explicitly.')
        citing,cited=match[0]
    else:citing,cited=col['citing'],col['cited']
    import pyarrow as pa
    import pyarrow.parquet as pq
    def id_expr(column):
        dtype=pq.ParquetFile(path).schema_arrow.field(column).type
        name=sql_identifier(column)
        if pa.types.is_floating(dtype):raise ValueError('Citation IDs cannot be floating point.')
        if pa.types.is_integer(dtype):return f'CAST({name} AS UBIGINT)'
        return "CAST(regexp_replace(trim(CAST("+name+" AS VARCHAR)), '^(https?://openalex[.]org/)?W', '') AS UBIGINT)"
    m=pd.read_parquet(d['data']/'papers.parquet',columns=['paper_id','year','AIProb'])
    assert_sample(m,c,'reference-cohort citing papers')
    con=analytical_db(c)
    con.register('selected',m)
    # The cited endpoint remains the original local reference universe so that
    # the graph display is compatible with the user's precomputed indicators.
    query=f'''WITH e AS (SELECT {id_expr(citing)} AS citing_id,{id_expr(cited)} AS cited_id
                          FROM read_parquet({sql_string(path)})),
       core AS (SELECT paper_id,year FROM read_parquet({sql_string(source(c,'papers'))})),
       scores AS (SELECT paper_id,AIProb FROM read_parquet({sql_string(source(c,'ai'))}))
       SELECT a.year AS citing_year,b.year AS cited_year,
          CAST(a.AIProb>0.5 AS TINYINT) AS ai_group,
          CASE WHEN r.AIProb IS NULL THEN 2 WHEN r.AIProb>0.5 THEN 1 ELSE 0 END AS reference_ai_group,
          COUNT(*) AS edges,CAST(a.year-b.year AS INTEGER) AS reference_age
       FROM e JOIN selected a ON e.citing_id=a.paper_id
              JOIN core b ON e.cited_id=b.paper_id
              LEFT JOIN scores r ON b.paper_id=r.paper_id
       WHERE e.citing_id<>e.cited_id AND b.year>=1 AND b.year=floor(b.year) AND b.year<=a.year
       GROUP BY ALL ORDER BY citing_year,cited_year,ai_group,reference_ai_group'''
    try:result=con.execute(query).fetch_df()
    finally:con.close()
    csv(result,d['tables']/'reference_cohorts.csv')
    return result
