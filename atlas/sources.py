"""Publishing-source definitions and fixed examples for the compact figures."""
import pandas as pd
from .runtime import exact_ids


def source_ids(series):
    """Preserve integer IDs, including uint64 identifiers above 2**53."""
    s = pd.Series(series, copy=False)
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype('string').str.strip().str.replace(
            r'^(https?://openalex[.]org/)?S', '', regex=True)
    return exact_ids(s)


def build_source_catalog(master, c):
    """Select at most six supported examples; retain every source in analyses."""
    recent = master[master.year.between(*c['comparison_years'])]
    s = recent.groupby('source_id', observed=True).AIProb.agg(
        recent_scored='size', recent_ai_positive=lambda x: int(x.gt(.5).sum()))
    all_counts = master.groupby('source_id', observed=True).AIProb.agg(
        all_scored='size', all_ai_positive=lambda x: int(x.gt(.5).sum()))
    s = all_counts.join(s).fillna(0).reset_index()
    counts = ['all_scored', 'all_ai_positive', 'recent_scored', 'recent_ai_positive']
    s[counts] = s[counts].astype('int64')
    s['recent_other_scored'] = s.recent_scored - s.recent_ai_positive
    s['example_eligible'] = (s.recent_ai_positive.ge(c['min_source_group_n'])
                            & s.recent_other_scored.ge(c['min_source_group_n']))
    s = s.sort_values(['recent_scored', 'source_id'], ascending=[False, True])
    requested = c.get('representative_source_ids', [])
    if requested:
        selected = source_ids(pd.Series(requested)).tolist()
        if len(selected) > 6 or len(set(selected)) != len(selected):
            raise ValueError('representative_source_ids must contain one to six distinct source IDs.')
        absent = set(selected) - set(s.source_id)
        if absent:
            raise ValueError(f'Configured example sources are absent: {sorted(absent)}')
        unsupported = set(selected) - set(s.loc[s.example_eligible, 'source_id'])
        if unsupported:
            raise ValueError(f'Example sources lack recent two-group support: {sorted(unsupported)}')
    else:
        selected = s.loc[s.example_eligible, 'source_id'].head(c['representative_source_n']).tolist()
    order = {int(value): i for i, value in enumerate(selected)}
    s['selected'] = s.source_id.isin(selected)
    s['display_order'] = s.source_id.map(order).astype('Int64')
    labels = {str(int(k)): str(v) for k, v in c.get('source_labels', {}).items()}
    s['source_label'] = [labels.get(str(int(v)), f'S{int(v)}') for v in s.source_id]
    if s.loc[s.selected, 'source_label'].duplicated().any():
        raise ValueError('Example source labels must be distinct.')
    return s.reset_index(drop=True)


def representative_sources(c):
    from .scope import read_table
    s = read_table(c, 'source_catalog.csv')
    return s.loc[s.selected].sort_values('display_order').reset_index(drop=True)
