from pathlib import Path
import json, os, gc, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT = dict(
    research_dir='data/SciSciNet2_research',
    output_dir='results_scored_sources', end_year=2024, trend_start=1990, ai_start=1980,
    batch_rows=1_000_000, max_rss_gib=384, physical_memory_fraction=.80,
    plot_sample_n=250_000, seed=20261031, r_export_fraction=1.0,
    baseline_years=[2010,2014], comparison_years=[2020,2024],
    min_country_period_mass=20, min_cell_period_mass=5,
    fixed_support_start=2000, min_ai_annual_mass=1, min_ai_period_mass=3,
    portfolio_window_years=5, standardization_window_years=5,
    min_profile_cell_n=5, min_surface_n=20, duckdb_memory_gib=16,
    source_labels={}, representative_source_ids=[], representative_source_n=6,
    min_source_group_n=20, edges_enabled=True,
    edge_columns={'citing':'citing_id','cited':'cited_id'},
    paths={
      'papers':'core/papers.parquet','ai':'extended/AI_papers.parquet',
      'members':'extended/country_members.parquet',
      'idr':'metrics/interdisciplinarity.parquet',
      'semantic':'metrics/embedding_paper.parquet',
      'history':'metrics/embedding_field_history.parquet',
      'edges':'stage/edges/data.parquet'})

def config(path=None):
    p=Path(path or ROOT/'config.json')
    c=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    out=DEFAULT | c
    out['paths']=DEFAULT['paths'] | c.get('paths',{})
    if out['end_year']!=2024:
        raise ValueError('This analysis is defined through 2024; adjust the design before changing end_year.')
    if not 0 < out['r_export_fraction'] <= 1: raise ValueError('r_export_fraction must be in (0,1].')
    if out['ai_start'] != 1980:
        raise ValueError('The scoring-corpus analysis starts in 1980.')
    if not 1 <= int(out['representative_source_n']) <= 6:
        raise ValueError('Choose one to six example sources to fit the six-row displays.')
    if float(out['min_source_group_n']) < 1:
        raise ValueError('min_source_group_n must be positive.')
    for key in ['min_ai_coverage', 'map_min_ai_coverage']:
        out.pop(key, None)
    return out

def dirs(c):
    p=Path(c['output_dir'])
    if not p.is_absolute(): p=ROOT/p
    ans={k:p/k for k in ['data','tables','figures','r','status']}
    for x in ans.values(): x.mkdir(parents=True,exist_ok=True)
    return ans

def fresh_outputs(c):
    """Start a new analysis without deleting source data or arbitrary folders."""
    import shutil
    p=Path(c['output_dir'])
    if not p.is_absolute():p=ROOT/p
    p=p.resolve()
    inputs=Path(c['research_dir']).resolve()
    if p==ROOT or p in ROOT.parents or p==inputs or p in inputs.parents or inputs in p.parents:
        raise ValueError('Use a dedicated output folder outside the source research directory.')
    marker=p/'ANALYSIS_OUTPUT.json'
    allowed={'data','tables','figures','r','status','ANALYSIS_OUTPUT.json'}
    if p.exists() and any(p.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get('project')!='JOI_scored_sources':
            raise ValueError('Choose an empty output folder; this folder was not created by this analysis.')
        unexpected={x.name for x in p.iterdir()}-allowed
        if unexpected:raise ValueError(f'Output folder contains unrelated files: {sorted(unexpected)}')
        for name in allowed-{'ANALYSIS_OUTPUT.json'}:
            child=p/name
            if child.is_symlink():raise ValueError('Output subdirectories cannot be symbolic links.')
        for name in allowed-{'ANALYSIS_OUTPUT.json'}:
            child=p/name
            if child.is_dir():shutil.rmtree(child)
            elif child.exists():child.unlink()
    p.mkdir(parents=True,exist_ok=True)
    json_write({'project':'JOI_scored_sources'},marker)
    return dirs(c)

def source(c,key):
    value = c['paths'].get(key)
    if not value:
        raise ValueError(f'Configure paths.{key} before using this optional input.')
    path = Path(value)
    return path if path.is_absolute() else Path(c['research_dir'])/path

def check_memory(c,label=''):
    try:rss=psutil.Process().memory_info().rss/2**30
    except (psutil.Error,OSError):
        import resource,sys
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss=peak/(2**30 if sys.platform=='darwin' else 2**20)
    try:physical=psutil.virtual_memory().total/2**30
    except (OSError,RuntimeError):physical=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/2**30
    cap=min(float(c['max_rss_gib']),physical*c['physical_memory_fraction'])
    if rss>cap:
        raise MemoryError(f'{label}: RSS={rss:.1f} GiB > {cap:.1f} GiB. Reduce batch_rows; keep separate notebook kernels closed.')
    return rss,cap

def json_write(value,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def csv(frame,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    manifest=Path(path).parent.parent/'status'/'sample_scope.json'
    if Path(path).parent.name=='tables' and manifest.is_file():
        frame=frame.copy()
        frame['scope_id']=json.loads(manifest.read_text(encoding='utf-8'))['scope_id']
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.15g')

def names(path): return pq.ParquetFile(path).schema_arrow.names

def batches(path,columns=None,size=1_000_000):
    pf=pq.ParquetFile(path)
    missing=set(columns or [])-set(pf.schema_arrow.names)
    if missing: raise KeyError(f'{path}: missing columns {sorted(missing)}; actual={pf.schema_arrow.names}')
    for b in pf.iter_batches(batch_size=size,columns=columns,use_threads=False):
        yield b.to_pandas()

def exact_ids(s):
    s=pd.Series(s,copy=False)
    if s.isna().any(): raise ValueError('Null paper identifiers are not allowed.')
    if pd.api.types.is_float_dtype(s):
        if ((s%1)!=0).any() or (s.abs()>=2**53).any():
            raise ValueError('Unsafe floating-point identifiers. Read original integer/text ID columns.')
    if not pd.api.types.is_numeric_dtype(s):
        s=s.astype('string').str.replace(r'^https?://openalex.org/W','',regex=True).str.replace(r'^W','',regex=True)
        if not s.str.fullmatch(r'\d+').all(): raise ValueError('Unrecognized paper identifier format.')
    if (pd.to_numeric(s,errors='raise')<=0).any():raise ValueError('Identifiers must be positive.')
    return s.astype('uint64').to_numpy()

def hash64(ids,seed=20261031):
    x=np.asarray(ids,dtype=np.uint64)^np.uint64(seed)
    with np.errstate(over='ignore'):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return x^(x>>np.uint64(31))

class SingleWriter:
    """One Parquet file, many row groups; validate footer and row count on close."""
    def __init__(self,path):
        self.path=Path(path)
        self.writer=None;self.schema=None;self.handle=None;self.closed=False;self.rows=0
        self.path.parent.mkdir(parents=True,exist_ok=True)
    def write(self,frame):
        if self.closed:raise RuntimeError('Cannot write to a closed Parquet writer.')
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if self.writer is None:
            manifest=self.path.parent.parent/'status'/'sample_scope.json'
            if manifest.is_file():
                metadata=dict(table.schema.metadata or {})
                metadata[b'analysis_scope_id']=json.loads(manifest.read_text(encoding='utf-8'))['scope_id'].encode()
                table=table.replace_schema_metadata(metadata)
            self.schema=table.schema
            self.handle=self.path.open('wb')
            self.writer=pq.ParquetWriter(self.handle,self.schema,compression='zstd',use_dictionary=True)
        else: table=table.cast(self.schema)
        if len(frame):self.writer.write_table(table,row_group_size=len(frame))
        self.rows+=len(frame)
    def close(self,success=True):
        if self.closed:return
        written=self.writer is not None
        if self.writer:self.writer.close()
        if self.handle:
            self.handle.flush();os.fsync(self.handle.fileno());self.handle.close()
        self.writer=None;self.closed=True
        if success and written:
            actual=pq.ParquetFile(self.path).metadata.num_rows
            if actual!=self.rows:raise IOError(f'{self.path}: wrote {self.rows} rows, footer has {actual}.')
        elif self.path.exists():self.path.unlink()
    def __enter__(self): return self
    def __exit__(self,typ,val,tb): self.close(typ is None)

def write_parquet(frame,path,chunk=1_000_000):
    if frame.empty:
        with SingleWriter(path) as w:w.write(frame)
        return
    with SingleWriter(path) as w:
        for i in range(0,len(frame),chunk):w.write(frame.iloc[i:i+chunk])

def require(c,files):
    absent=[str(source(c,k)) for k in files if not source(c,k).is_file()]
    if absent: raise FileNotFoundError('Required local files:\n'+'\n'.join(absent))

def safe_ratio(a,b):
    aa,bb=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    return np.divide(aa,bb,out=np.full(aa.shape,np.nan),where=bb>0)

def run_stage(c,label,fn):
    t=time.monotonic();print(f'{label}: start',flush=True)
    result=fn(c)
    print(f'{label}: complete in {(time.monotonic()-t)/60:.1f} min',flush=True)
    gc.collect();return result

def sql_string(value):
    return "'"+str(value).replace("'", "''")+"'"

def sql_identifier(value):
    return '"'+str(value).replace('"','""')+'"'

def analytical_db(c):
    import duckdb
    con=duckdb.connect()
    con.execute(f"SET memory_limit={sql_string(str(float(c.get('duckdb_memory_gib',16)))+'GB')}")
    con.execute('SET threads=4')
    con.execute(f"SET temp_directory={sql_string(dirs(c)['data']/'duckdb_temp')}")
    con.execute('SET preserve_insertion_order=false')
    return con

def query_batches(con,query,size):
    relation=con.sql(query)
    if hasattr(relation,'to_arrow_reader'):
        return relation.to_arrow_reader(batch_size=size)
    result=con.execute(query)
    if hasattr(result,'to_arrow_reader'):
        return result.to_arrow_reader(batch_size=size)
    return result.fetch_record_batch(rows_per_batch=size)
