"""Single sources for manuscript Tables 7/S13 and Figure panels 5d/3e."""
import numpy as np
import pandas as pd
from .runtime import dirs,csv
from .scope import read_table,require_scope
from .labels import GROUP_LABELS

REPORT_METRICS = ['known','ai','nonai','score']
LABELS = GROUP_LABELS


def report_frames(concentration, standardization, year=2024):
    needed = {'year','metric','n_countries','gini','hhi','top5_share','top10_share'}
    if not needed.issubset(concentration):
        raise ValueError(f'Concentration columns missing: {needed-set(concentration)}')
    table7 = concentration[concentration.year.eq(year)&concentration.metric.isin(REPORT_METRICS)].copy()
    if table7.metric.duplicated().any():
        raise ValueError('Table 7 needs exactly one row per year and publication mass.')
    if set(table7.metric) != set(REPORT_METRICS):
        raise ValueError(f'No complete {year} concentration summary; never substitute a five-year total.')
    table7 = table7.set_index('metric').loc[REPORT_METRICS].reset_index()
    table7['publication_mass'] = table7.metric.map(LABELS)
    table7['top_five_pct'] = table7.top5_share*100
    table7['top_ten_pct'] = table7.top10_share*100
    if not table7.top_five_pct.between(0,100).all():
        raise ValueError('Top-country shares must lie on the 0–100 percent scale.')
    columns=['year','country','n_sources','retained_baseline_source_mass','raw_common_share','standardized_share']
    if not set(columns).issubset(standardization):
        raise ValueError(f'Standardization columns missing: {set(columns)-set(standardization)}')
    if 'country_all_scored_share' in standardization:
        columns.append('country_all_scored_share')
    table13 = standardization.loc[standardization.year.eq(year),columns].copy()
    if table13.country.duplicated().any():
        raise ValueError('Table S13 / Figure 3e needs one point per country in 2024.')
    table13 = table13.sort_values('country').reset_index(drop=True)
    for col in ['raw_common_share','standardized_share','retained_baseline_source_mass']:
        if not table13[col].between(0,1).all():
            raise ValueError(f'{col} must be a finite zero-to-one share.')
    table13['raw_ai_positive_pct'] = table13.raw_common_share*100
    table13['standardized_ai_positive_pct'] = table13.standardized_share*100
    table13['baseline_source_mass_retained_pct'] = table13.retained_baseline_source_mass*100
    table13['rate_counting_unit']='Unique-paper country-source credit'
    if 'country_all_scored_share' in table13:
        if not table13.country_all_scored_share.between(0,1).all():
            raise ValueError('The independent unique-paper country AI rate must be finite in [0,1].')
        table13['country_all_scored_ai_pct']=table13.country_all_scored_share*100
    bars = table7[table7.metric.isin(['known','ai','nonai'])].copy()
    points = table13.copy()
    return table7,table13,bars,points


def export_publication_tables(c):
    require_scope(c)
    frames = report_frames(read_table(c,'country_concentration_summary.csv'),
        read_table(c,'ai_standardized_common_support.csv'),c['end_year'])
    names = ['Table7_country_concentration','TableS13_country_standardization',
             'Fig5d_source','Fig3e_source']
    for name,frame in zip(names,frames):
        csv(frame,dirs(c)['tables']/f'{name}_{c["end_year"]}.csv')
    # Formatting is a separate view; plots always use the full-precision sources.
    a,b,_,_ = frames
    publication7 = a[['publication_mass','n_countries','gini','hhi','top_five_pct','top_ten_pct']].copy()
    for key in ['gini','hhi']:publication7[key]=publication7[key].map(lambda x:f'{x:.3f}')
    for key in ['top_five_pct','top_ten_pct']:publication7[key]=publication7[key].map(lambda x:f'{x:.2f}')
    columns13=['country','n_sources','baseline_source_mass_retained_pct','raw_ai_positive_pct','standardized_ai_positive_pct']
    if 'country_all_scored_ai_pct' in b:columns13.append('country_all_scored_ai_pct')
    publication13 = b[columns13].copy()
    for key in publication13.columns[2:]:publication13[key]=publication13[key].map(lambda x:f'{x:.2f}')
    publication13=publication13.rename(columns={'country':'Country','n_sources':'Retained source count',
        'baseline_source_mass_retained_pct':'Retained benchmark source credit (%)',
        'raw_ai_positive_pct':'Raw AI-positive source share (%)',
        'standardized_ai_positive_pct':'Source-adjusted AI-positive source share (%)',
        'country_all_scored_ai_pct':'Unique-paper AI-positive share (%)'})
    publication7.to_csv(dirs(c)['tables']/f'Table7_publication_{c["end_year"]}.csv',index=False,encoding='utf-8-sig')
    publication13.to_csv(dirs(c)['tables']/f'TableS13_publication_{c["end_year"]}.csv',index=False,encoding='utf-8-sig')
    return frames


def export_profile_support_tables(c, contrasts):
    """Table S11 states the exact comparison sources for every metric and year."""
    labels={
        'ref_field_simpson':('Reference-field Simpson diversity','Simpson units',3),
        'emb_ref_pair_mean_distance':('Reference-pair semantic distance','Cosine-distance units',4),
        'emb_field_history_distance':('Historical field distance','Cosine-distance units',4),
        'mean_ref_age':('Mean reference age','Years',2),
    }
    t=contrasts[contrasts.metric.isin(labels)].copy()
    t['indicator']=t.metric.map(lambda x:labels[x][0])
    t['unit']=t.metric.map(lambda x:labels[x][1])
    t['counting_unit']='Unique papers within supported publishing sources'
    t=t.sort_values(['year','metric']).reset_index(drop=True)
    csv(t,dirs(c)['tables']/'TableS11_profile_comparisons.csv')
    p=t[['year','indicator','source_adjusted_difference','unit','n_sources','retained_sources',
         'comparison_set','support_changed','n0','n1']].copy()
    def format_difference(metric,value):
        if pd.isna(value):return '—'
        digits=labels[metric][2]
        text=f'{value:.{digits}f}'
        return f'{0:.{digits}f}' if float(text)==0 else text
    p['source_adjusted_difference']=[format_difference(m,v)
        for m,v in zip(t.metric,t.source_adjusted_difference)]
    p['retained_sources']=p.retained_sources.replace('','None')
    p=p.rename(columns={'year':'Year','indicator':'Indicator',
        'source_adjusted_difference':'Source-adjusted difference (AI-positive minus Other-scored)',
        'unit':'Original unit','n_sources':'Retained source count','retained_sources':'Retained sources',
        'comparison_set':'Comparison support','support_changed':'Source set changed from previous year',
        'n0':'Other-scored papers','n1':'AI-positive papers'})
    p.to_csv(dirs(c)['tables']/'TableS11_publication.csv',index=False,encoding='utf-8-sig')
    return t
