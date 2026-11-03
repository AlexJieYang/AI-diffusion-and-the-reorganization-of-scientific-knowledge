"""A common score-observed paper population for Python and R."""
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CONTRACT_VERSION = 1
SCOPE_RULE = 'observed_ai_score__source_venues__1980_2024'


def assert_sample(frame, c, context='analytical data'):
    required = {'AIProb', 'year'}
    if not required.issubset(frame):
        raise ValueError(f'{context}: missing sample columns {required-set(frame)}')
    score = pd.to_numeric(frame.AIProb, errors='coerce').to_numpy(float)
    if not (np.isfinite(score) & (score >= 0) & (score <= 1)).all():
        raise ValueError(f'{context}: only observed, valid AI scores are eligible.')
    year = pd.to_numeric(frame.year, errors='coerce')
    if not (year.between(c['ai_start'], c['end_year']) & year.mod(1).eq(0)).all():
        raise ValueError(f'{context}: invalid publication years.')


def source_settings(c):
    return {k: c[k] for k in ['source_labels', 'representative_source_ids',
        'representative_source_n', 'min_source_group_n', 'comparison_years']}


def record_scope(c, frame, flow):
    from .runtime import dirs, json_write, source
    assert_sample(frame, c, 'master')
    order = np.argsort(frame.paper_id.to_numpy())
    digest = hashlib.sha256()
    for column, dtype in [('paper_id', '<u8'), ('year', '<i2'),
                          ('source_id', '<u8'), ('AIProb', '<f8')]:
        digest.update(frame[column].to_numpy()[order].astype(dtype).tobytes())
    signatures = {}
    for key in ['papers', 'ai', 'members', 'idr', 'semantic', 'history']:
        path = source(c, key)
        stat = path.stat()
        signatures[key] = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    settings = dict(contract_version=CONTRACT_VERSION, scope_rule=SCOPE_RULE,
        source_settings=source_settings(c), ai_start=c['ai_start'],
        end_year=c['end_year'], input_signatures=signatures)
    digest.update(json.dumps(settings, sort_keys=True).encode())
    manifest = settings | dict(scope_id=digest.hexdigest(), n_papers=len(frame),
        n_ai_positive=int(frame.AIProb.gt(.5).sum()), sample_flow=flow,
        counting_unit='One paper, one publishing source; countries share one paper credit.',
        metric_scope='Existing local reference-diversity, semantic, historical-centroid and citation values are joined unchanged.')
    json_write(manifest, dirs(c)['status']/'sample_scope.json')
    return manifest


def require_scope(c, check_master=True):
    from .runtime import dirs
    path = dirs(c)['status']/'sample_scope.json'
    if not path.is_file():
        raise ValueError('No analytical population. Run from the master stage.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('contract_version') != CONTRACT_VERSION
            or manifest.get('scope_rule') != SCOPE_RULE
            or manifest.get('source_settings') != source_settings(c)
            or manifest.get('ai_start') != c['ai_start']
            or manifest.get('end_year') != c['end_year']):
        raise ValueError('Outputs do not match this source-based configuration; run from scratch.')
    if check_master:
        require_parquet(dirs(c)['data']/'papers.parquet', manifest['scope_id'])
    return manifest


def require_parquet(path, scope_id):
    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    if meta.get(b'analysis_scope_id', b'').decode() != scope_id:
        raise ValueError(f'{path}: output belongs to another analytical population.')


def read_table(c, name, optional=False):
    from .runtime import dirs
    manifest = require_scope(c)
    path = dirs(c)['tables']/name
    if not path.is_file():
        if optional:
            return pd.DataFrame()
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if 'scope_id' not in frame or not frame.scope_id.eq(manifest['scope_id']).all():
        raise ValueError(f'{path}: table does not match the current paper population.')
    return frame.drop(columns='scope_id')
