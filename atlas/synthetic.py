"""Small, explicitly synthetic fixtures for functional checks; never research evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from .runtime import DEFAULT,write_parquet

def make_fixture(root,n=24000):
    root=Path(root);rng=np.random.default_rng(283)
    for folder in ['core','metrics','extended','stage/edges']:(root/folder).mkdir(parents=True,exist_ok=True)
    ids=np.arange(2**53+1,2**53+1+n,dtype='uint64')
    y=rng.integers(1990,2025,n,dtype='int16');y[:7]=[1750,1890,1979,1980,0,2025,2019]
    prob=rng.beta(.7,2,n).astype('float64');prob[rng.random(n)<.12]=np.nan
    prob[10:14]=[0,.5,1,np.nan]
    prob[14:16]=[np.nextafter(.5,1),np.nextafter(.5,0)]
    y[10:16]=2019
    y[18:21]=2024;prob[18:21]=[.9,.8,.7]
    ref=rng.integers(0,100,n).astype('float32');team=rng.integers(1,25,n).astype('float32')
    base=pd.DataFrame({'paper_id':ids,'year':y,'source_id':rng.integers(1,13,n,dtype='uint64'),
       'is_conference':rng.random(n)<.15,'n_references':ref,
       'mean_ref_age':rng.gamma(3,4,n),'team_size':team,'n_countries':rng.integers(0,4,n),
       'is_international':rng.random(n)<.25,'mean_geo_km':rng.uniform(0,12000,n),
       'c5':rng.poisson(8,n).astype(float),'c5_norm':rng.exponential(1,n),'c5_pct':rng.random(n),'top10_credit':(rng.random(n)<.1).astype(float),
       'mean_prior_papers':rng.gamma(2,5,n)})
    base.loc[18:20,'n_countries']=[2,1,1]
    write_parquet(base,root/'core/papers.parquet',1000)
    write_parquet(pd.DataFrame({'paper_id':ids,'AIProb':prob}),root/'extended/AI_papers.parquet',997)
    sim=np.clip(.35+.12*np.nan_to_num(prob)+rng.normal(0,.13,n),0,1)
    nfield=np.minimum(ref,rng.integers(0,100,n))
    idr=pd.DataFrame({'paper_id':ids,'n_field_refs':nfield,'ref_field_coverage':np.divide(nfield,ref,out=np.zeros(n),where=ref>0),
       'ref_field_variety':rng.integers(1,6,n),'ref_field_shannon':rng.uniform(0,2,n),
       'ref_field_simpson':sim,'ref_field_rao':rng.uniform(0,.6,n),'ref_outside_primary_share':rng.random(n)})
    write_parquet(idr,root/'metrics/interdisciplinarity.parquet',1200)
    embcov=rng.random(n)
    emb=pd.DataFrame({'paper_id':ids,'emb_ref_coverage':embcov,'emb_ref_pair_mean_distance':rng.uniform(.05,.6,n),
                      'emb_ref_mean_distance':rng.random(n),'emb_ref_centroid_distance':rng.random(n)})
    write_parquet(emb,root/'metrics/embedding_paper.parquet',1200)
    hist=pd.DataFrame({'paper_id':ids,'emb_field_history_count':rng.integers(10,100,n),'emb_field_history_distance':rng.uniform(0,.5,n)})
    write_parquet(hist,root/'metrics/embedding_field_history.parquet',1200)
    codes=np.array(['US','CN','GB','DE','JP','BR']);cc=rng.integers(0,len(codes),n)
    member=[]
    for j in range(n):
        if base.n_countries.iloc[j]==0:continue
        cs=[codes[cc[j]]]
        if base.n_countries.iloc[j]>1:cs.append(codes[(cc[j]+1)%len(codes)])
        for code in cs:member.append((ids[j],int(y[j]),code,1/len(cs)))
    write_parquet(pd.DataFrame(member,columns=['paper_id','year','entity_id','weight']),root/'extended/country_members.parquet',1100)
    edges=[]
    for j in range(n):
        candidates=np.flatnonzero((y<=y[j])&(y>0)&(np.arange(n)!=j))
        if len(candidates):
            for k in rng.choice(candidates,size=min(6,len(candidates)),replace=False):edges.append((ids[j],ids[k]))
    write_parquet(pd.DataFrame(edges,columns=['citing_id','cited_id']),root/'stage/edges/data.parquet',1000)
    ky=[]
    for year in range(1990,2025):
      for country in codes:
        ky.append(dict(year=year,country_code=country,knowledge_input=1000,external_knowledge_input=rng.uniform(100,900)))
    write_parquet(pd.DataFrame(ky),root/'extended/country_knowledge_year.parquet')
    return root

def fixture_config(root,output):
    c=DEFAULT.copy();c['paths']=DEFAULT['paths'].copy()
    c.update(research_dir=str(Path(root).resolve()),output_dir=str(Path(output).resolve()),
             batch_rows=3000,plot_sample_n=3000,min_country_period_mass=5,min_cell_period_mass=2,
             min_surface_n=3,fixed_support_start=2024,min_ai_annual_mass=1,demo=True,
             min_profile_cell_n=2,min_source_group_n=1,duckdb_memory_gib=1,portfolio_window_years=1,
             standardization_window_years=1,min_ai_period_mass=1)
    return c
