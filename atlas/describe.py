import gc
import numpy as np
import pandas as pd
from .scope import require_scope, assert_sample,read_table
from .runtime import dirs,batches,csv,check_memory,hash64,SingleWriter,write_parquet,safe_ratio,json_write

METRIC_BINS={
 'AIProb':np.linspace(0,1,101),
 'ref_field_simpson':np.linspace(0,1,101),
 'ref_field_shannon':np.linspace(0,4,101),
 'emb_ref_pair_mean_distance':np.linspace(0,2,101),
 'emb_field_history_distance':np.linspace(0,2,101),
 'mean_ref_age':np.arange(0,102,2),
 'team_size':np.r_[0,1,2,3,4,5,7,10,15,20,30,50,100,200,500],
 'n_references':np.r_[0,1,2,5,10,20,40,80,160,320,640,1280]}
MEASURES=list(METRIC_BINS)+['is_international','emb_ref_coverage',
                         'country_author_coverage','c5_norm']
PROFILE_MEASURES=['ref_field_simpson','emb_ref_pair_mean_distance',
                  'emb_field_history_distance','mean_ref_age','c5_norm']

def refresh_surface_support(c):
    """Apply the display threshold to cached full-population bin summaries."""
    surf=read_table(c,'response_surface.csv')
    surf['supported']=surf.n.ge(c['min_surface_n'])
    csv(surf,dirs(c)['tables']/'response_surface.csv')
    return surf

def group_codes(prob):
    prob=np.asarray(prob,dtype=float)
    if not (np.isfinite(prob)&(prob>=0)&(prob<=1)).all():
        raise ValueError('Only observed valid AI scores may enter a comparison group.')
    return (prob>.5).astype('int8')

def merge_sums(parts,keys):
    if not parts:return pd.DataFrame(columns=keys)
    return pd.concat(parts,ignore_index=True).groupby(keys,observed=True,dropna=False).sum(numeric_only=True).reset_index()

def summarize_frame(b,keys,measures=MEASURES):
    z=b[keys].copy();z['n']=np.ones(len(b),dtype='int64')
    for x in measures:
        v=b[x].to_numpy(dtype='float64');ok=np.isfinite(v)
        z[x+'_n']=ok.astype('int64');z[x+'_sum']=np.where(ok,v,0)
        z[x+'_ss']=np.where(ok,v*v,0)
    return z.groupby(keys,observed=True,dropna=False).sum(numeric_only=True).reset_index()

def finalize(s,measures=MEASURES):
    z=s.copy()
    for x in measures:
        n=z[x+'_n'];mean=safe_ratio(z[x+'_sum'],n)
        z[x+'_mean']=mean
        z[x+'_sd']=np.sqrt(np.maximum(0,safe_ratio(z[x+'_ss'],n)-mean**2))
        z[x+'_coverage']=safe_ratio(n,z.n)
    return z

def build_descriptions(c):
    manifest=require_scope(c)
    d=dirs(c);path=d['data']/'papers.parquet'
    sums=[];source_sums=[];hist=[];surface=[];reservoir={};population=np.zeros(2,dtype='int64')
    target=max(10,int(c['plot_sample_n'])//2)
    refs=np.array([0,2,5,10,20,40,80,160,320],float)
    rcols=['paper_id','year','source_id','is_conference','AIProb','n_references','team_size',
           'ref_field_simpson','emb_ref_pair_mean_distance','emb_field_history_distance','mean_ref_age','c5_norm']
    with SingleWriter(d['data']/'r_papers.parquet') as rw:
      for batch_no,b in enumerate(batches(path,size=c['batch_rows'])):
        assert_sample(b,c,'description batch')
        # Eligibility for a mature citation comparison; source values remain
        # untouched in the analytical master and are never renormalized.
        b.loc[b.year.gt(c['end_year']-5),'c5_norm']=np.nan
        b['ai_group']=group_codes(b.AIProb.to_numpy());keys=['year','ai_group']
        sums.append(summarize_frame(b,keys))
        if len(sums)>=10:sums=[merge_sums(sums,keys)]
        source_sums.append(summarize_frame(b,['year','source_id','ai_group'],PROFILE_MEASURES))
        if len(source_sums)>=10:
            source_sums=[merge_sums(source_sums,['year','source_id','ai_group'])]
        # Exact histogram counts, including under/overflow; no sampled trend estimates.
        for (year,group),g in b.groupby(['year','ai_group'],observed=True):
          if year<c['trend_start']:continue
          for x,edges in METRIC_BINS.items():
            v=g[x].to_numpy(dtype=float);v=v[np.isfinite(v)]
            counts,_=np.histogram(v,bins=edges)
            hh=pd.DataFrame({'year':int(year),'ai_group':int(group),'metric':x,
                             'bin':np.arange(len(counts)),'count':counts})
            hh['under']=0;hh['over']=0
            hh.loc[0,'under']=int((v<edges[0]).sum());hh.loc[len(hh)-1,'over']=int((v>edges[-1]).sum())
            hist.append(hh)
        if len(hist)>500:hist=[merge_sums(hist,['year','ai_group','metric','bin'])]
        h=hash64(b.paper_id.to_numpy(),c['seed'])
        for group in range(2):
            use=np.flatnonzero(b.ai_group.to_numpy()==group);population[group]+=len(use)
            if len(use)>target:use=use[np.argpartition(h[use],target-1)[:target]]
            candidate=b.iloc[use].copy();candidate['_hash']=h[use]
            if group in reservoir:candidate=pd.concat([reservoir[group],candidate],ignore_index=True)
            if len(candidate)>target:
                ix=np.argpartition(candidate._hash.to_numpy(),target-1)[:target];candidate=candidate.iloc[ix]
            reservoir[group]=candidate
        use=b.AIProb.notna()&b.year.ge(c['ai_start'])
        if c['r_export_fraction']<1:
            use &= (h.astype('float64')/float(2**64))<c['r_export_fraction']
        rr=b.loc[use,rcols].copy();rr['sampling_weight']=1/c['r_export_fraction'];rr['scope_id']=manifest['scope_id'];rw.write(rr)
        # A supported descriptive surface. References >=320 are counted separately.
        z=b.loc[b.AIProb.notna()&b.year.ge(c['trend_start']),['AIProb','n_references','ref_field_simpson']].copy()
        z=z.loc[z.n_references.ge(0)&z.n_references.lt(320)&z.ref_field_simpson.notna()]
        if len(z):
            z['ai_bin']=np.minimum((z.AIProb*10).astype(int),9)
            z['ref_bin']=np.searchsorted(refs,z.n_references,side='right')-1
            z['n']=1;z['value_sum']=z.ref_field_simpson.astype('float64')
            surface.append(z.groupby(['ai_bin','ref_bin'])[['n','value_sum']].sum().reset_index())
        if batch_no%10==0:print(f'Descriptions: {(batch_no+1)*c["batch_rows"]:,} input rows scanned',flush=True)
        check_memory(c,'descriptions')
    # Publishing sources partition the unique-paper population exactly.
    annual=merge_sums(sums,['year','ai_group'])
    csv(finalize(annual),d['tables']/'annual_groups.csv')
    allannual=merge_sums([annual.drop(columns='ai_group')],['year'])
    csv(finalize(allannual),d['tables']/'annual_all.csv')
    s=merge_sums(source_sums,['year','source_id','ai_group'])
    csv(finalize(s,PROFILE_MEASURES),d['tables']/'source_year_groups.csv')
    h=merge_sums(hist,['year','ai_group','metric','bin'])
    for x,edges in METRIC_BINS.items():
        use=h.metric.eq(x);binidx=h.loc[use,'bin'].to_numpy(dtype=int)
        h.loc[use,'left']=edges[binidx];h.loc[use,'right']=edges[binidx+1]
    csv(h,d['tables']/'histograms.csv')
    qs=[]
    for (y,g,x),hh in h.groupby(['year','ai_group','metric'],observed=True):
        hh=hh.sort_values('bin');n=int(hh['count'].sum()+hh.under.sum()+hh.over.sum())
        if not n:continue
        cumulative=hh['count'].cumsum().to_numpy()+hh.under.sum()
        for q in [.1,.25,.5,.75,.9]:
            j=np.searchsorted(cumulative,q*n)
            lo=float(hh.iloc[j].left) if j<len(hh) else np.inf
            hi=float(hh.iloc[j].right) if j<len(hh) else np.inf
            if q*n<=hh.under.sum():lo=-np.inf;hi=float(hh.left.min())
            qs.append(dict(year=y,ai_group=g,metric=x,quantile=q,lower=lo,upper=hi,n=n))
    csv(pd.DataFrame(qs),d['tables']/'quantile_intervals.csv')
    sample=[]
    for group,z in reservoir.items():
        z=z.drop(columns='_hash');z['sample_weight']=population[group]/max(1,len(z));sample.append(z)
    write_parquet(pd.concat(sample,ignore_index=True),d['data']/'plot_sample.parquet')
    surf=merge_sums(surface,['ai_bin','ref_bin'])
    if surf.empty:
        surf=pd.DataFrame(columns=['ai_bin','ref_bin','n','value_sum'])
    surf['mean']=safe_ratio(surf.value_sum,surf.n)
    surf['supported']=surf.n.ge(c['min_surface_n'])
    csv(surf,d['tables']/'response_surface.csv')
    json_write(dict(plot_population_by_group=population.tolist(),plot_sample_by_group={str(k):len(v) for k,v in reservoir.items()},
                    histogram_quantiles='Intervals from full-population fixed bins; not exact point quantiles.',
                    r_export_fraction=c['r_export_fraction'],surface_reference_range='0<=references<320; wider tails remain in other summaries.'),
               d['status']/'description_scope.json')
    return finalize(allannual)
