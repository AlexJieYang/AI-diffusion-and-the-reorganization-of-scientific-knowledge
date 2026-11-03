from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from scipy.stats import gaussian_kde
from .runtime import ROOT,dirs,csv,safe_ratio,json_write
from .scope import read_table,require_scope,require_parquet
from .publication_tables import export_publication_tables
from .sources import representative_sources
from .labels import GROUP_LABELS

BLUE='#0072B2';ORANGE='#D55E00';TEAL='#009E73';PURPLE='#8064A2';GRAY='#8C969D'
GROUPS={0:(GROUP_LABELS['nonai'],BLUE),1:(GROUP_LABELS['ai'],ORANGE)}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.5,'axes.titlesize':7,
      'axes.labelsize':6.5,'xtick.labelsize':5.5,'ytick.labelsize':5.5,'legend.fontsize':5.4,
      'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55,
      'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':450,'lines.linewidth':1.15})

def start(c,height=160):
    """Compact canvas; figure titles and explanatory notes live in captions."""
    fig=plt.figure(figsize=(180/25.4,height/25.4),layout='constrained')
    fig.get_layout_engine().set(w_pad=.035,h_pad=.035,wspace=.04,hspace=.04)
    if c.get('demo'):
        # Keep synthetic output visibly identified during functional checks.
        fig.suptitle('SYNTHETIC DEMONSTRATION - NO RESEARCH RESULTS',
                     fontsize=6,color=ORANGE)
    return fig

def lab(ax,letter):
    """Retain panel letters without drawing panel titles, including in 3D."""
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(0,4),
                textcoords='offset points',fontsize=8,fontweight='bold',
                ha='left',va='bottom',annotation_clip=False)

def no_data(ax,text='Insufficient observed support'):
    ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes,color=GRAY,fontsize=6,wrap=True)
    ax.set_xticks([]);ax.set_yticks([])

def read(c,name):
    return read_table(c,name,optional=True)

def save(fig,c,name,caption):
    out=dirs(c)['figures']
    if any(ax.name=='3d' for ax in fig.axes):
        # Initialize projected tick/label positions before solving the layout.
        engine=fig.get_layout_engine()
        fig.set_layout_engine('none')
        fig.canvas.draw()
        fig.set_layout_engine(engine)
    # Tight export removes unused outer margins while retaining labels/legends.
    for ext in ['pdf','svg','png']:
        temporary=out/(name+'.temporary.'+ext)
        try:
            fig.savefig(temporary,facecolor='white',dpi=450,
                        bbox_inches='tight',pad_inches=.035)
            if ext=='png':
                from PIL import Image
                with Image.open(temporary) as raster:
                    raster.verify()
            temporary.replace(out/(name+'.'+ext))
        finally:
            temporary.unlink(missing_ok=True)
    (out/(name+'_caption.txt')).write_text(caption,encoding='utf-8');plt.close(fig)

def weighted_density(x,grid,weights=None):
    x=np.asarray(x,float);ok=np.isfinite(x);x=x[ok]
    if weights is not None:weights=np.asarray(weights,float)[ok]
    if len(x)<4 or np.ptp(x)<1e-10:return np.zeros_like(grid)
    return gaussian_kde(x,weights=weights)(grid)

def rain(ax,sample,metric,xlabel,years=(2015,2024)):
    z=sample[sample.year.between(*years)&sample.ai_group.isin([0,1])]
    allx=z[metric].dropna()
    if len(allx)<5:no_data(ax);return
    lo,hi=np.quantile(allx,[.001,.999]);lo=min(lo,0);hi=max(hi,lo+.01);grid=np.linspace(lo,hi,160)
    for j,g in enumerate([0,1]):
        v=z.loc[z.ai_group.eq(g),metric].dropna().to_numpy();den=weighted_density(v,grid)
        if den.max()>0:den=den/den.max()*.32
        col=GROUPS[g][1];ax.fill_between(grid,j+.03,j+.03+den,color=col,alpha=.65)
        if len(v):
            step=max(1,len(v)//600);vv=v[::step]
            jitter=np.random.default_rng(19+g).uniform(.05,.23,len(vv))
            ax.scatter(vv,j-jitter,s=1.3,color=col,alpha=.16,rasterized=True)
            q=np.quantile(v,[.25,.5,.75]);ax.plot(q[[0,2]],[j,j],lw=2,color=col);ax.plot(q[1],j,'o',ms=2,mec=col,mfc='white')
    ax.set_yticks([0,1],[GROUPS[0][0],GROUPS[1][0]]);ax.set_xlabel(xlabel)

def map_values(ax,values,label,vmin=None,vmax=None):
    data=json.loads((ROOT/'assets/world.geojson').read_text())
    v=np.array(list(values.values()),float);v=v[np.isfinite(v)]
    if not len(v):no_data(ax);return None
    norm=Normalize(vmin if vmin is not None else float(np.min(v)),vmax if vmax is not None else max(float(np.max(v)),float(np.min(v))+1e-6))
    cmap=plt.get_cmap('viridis')
    for f in data['features']:
        code=f['properties'].get('iso_a2');value=values.get(code,np.nan)
        col=cmap(norm(value)) if np.isfinite(value) else '#E2E6E9'
        geom=f['geometry'];polys=geom['coordinates'] if geom['type']=='MultiPolygon' else [geom['coordinates']]
        for poly in polys:
            xy=np.asarray(poly[0],float)
            if np.ptp(xy[:,0])>350:continue
            ax.fill(np.deg2rad(xy[:,0]),np.deg2rad(xy[:,1]),color=col,ec='white',lw=.12,zorder=2)
    ax.grid(alpha=.18,lw=.35);ax.set_xticklabels([]);ax.set_yticklabels([])
    m=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    cb=ax.figure.colorbar(m,ax=ax,orientation='horizontal',fraction=.045,pad=.08,aspect=35)
    cb.set_label(label,fontsize=5.5);cb.ax.tick_params(labelsize=5)
    return m

def heat(ax,pivot,cmap='viridis',vmin=None,vmax=None,labels=False):
    if pivot.empty:no_data(ax);return
    arr=np.ma.masked_invalid(pivot.to_numpy(float));cm=plt.get_cmap(cmap).copy();cm.set_bad('#E4E8EB')
    im=ax.imshow(arr,aspect='auto',origin='upper',interpolation='nearest',cmap=cm,vmin=vmin,vmax=vmax)
    if labels:
        ax.set_yticks(np.arange(len(pivot)),pivot.index.astype(str));ax.set_xticks(np.arange(len(pivot.columns)),pivot.columns.astype(str),rotation=90)
    return im

def alluvial(ax,states,periods):
    if states.empty:no_data(ax);return
    tab=states.pivot(index=['country','source_id'],columns='period',values='state').reindex(columns=periods).fillna('Unobserved / low support')
    order=['AI-positive specialized','Both specialized','Other-scored specialized','Neither specialized','Unobserved / low support']
    colors=[ORANGE,TEAL,BLUE,PURPLE,'#C9CED2'];n=len(tab)
    if not n:no_data(ax);return
    positions={}
    for k,year in enumerate(periods):
        count=tab[year].value_counts();cursor=0
        for state,col in zip(order,colors):
            h=count.get(state,0)/n;positions[(k,state)]=[cursor,cursor+h]
            ax.fill_between([k-.045,k+.045],[cursor,cursor],[cursor+h,cursor+h],color=col,lw=0)
            cursor+=h
    for k in range(len(periods)-1):
        counts=tab.groupby([periods[k],periods[k+1]]).size()
        left={s:positions[(k,s)][0] for s in order};right={s:positions[(k+1,s)][0] for s in order}
        for a,col in zip(order,colors):
          for b in order:
            h=counts.get((a,b),0)/n
            if not h:continue
            l=left[a];r=right[b];left[a]+=h;right[b]+=h;x0=k+.045;x1=k+1-.045
            verts=[(x0,l),(x0+.4,l),(x1-.4,r),(x1,r),(x1,r+h),
                   (x1-.4,r+h),(x0+.4,l+h),(x0,l+h),(x0,l)]
            codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,
                   MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts,codes),fc=col,ec='none',alpha=.38))
    ax.set_xlim(-.1,len(periods)-.9);ax.set_ylim(0,1);ax.set_xticks(range(len(periods)),[f'{y}-{str(y+4)[-2:]}' for y in periods]);ax.set_ylabel('Fraction of fixed units')
    for state,col in zip(order,colors):ax.plot([],[],lw=3,color=col,label=state)
    ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=5)

def figure1(c):
    a=read(c,'annual_all.csv');groups=read(c,'annual_groups.csv')
    a=a[a.year.between(c['trend_start'],c['end_year'])]
    raw=pd.read_parquet(dirs(c)['data']/'country_masses.parquet')
    recent=raw[raw.year.between(*c['comparison_years'])].groupby('country')[['known','ai']].sum()
    recent['share']=safe_ratio(recent.ai,recent.known)
    fig=start(c,height=175)
    gs=fig.add_gridspec(3,1,height_ratios=[1.15,1,1])
    top=gs[0].subgridspec(1,2,width_ratios=[2,1]);middle=gs[1].subgridspec(1,3);bottom=gs[2].subgridspec(1,2)
    aa=fig.add_subplot(top[0],projection='mollweide')
    counts=top[1].subgridspec(2,1,hspace=.12)
    bb=fig.add_subplot(counts[0]);bb_ai=fig.add_subplot(counts[1],sharex=bb)
    bb.set_gid('Fig1b_AllScored');bb_ai.set_gid('Fig1b_AIpositive')
    cc=fig.add_subplot(middle[0]);dd=fig.add_subplot(middle[1]);ee=fig.add_subplot(middle[2])
    ff=fig.add_subplot(bottom[0]);gg=fig.add_subplot(bottom[1])
    good=recent.known.ge(c['min_country_period_mass'])&recent.known.gt(0)
    map_values(aa,(100*recent.loc[good,'share']).to_dict(),'AI-positive share of scored output (%)');lab(aa,'a')
    positive=groups[groups.ai_group.eq(1)].set_index('year').n.reindex(a.year,fill_value=0)
    bb.plot(a.year,a.n/1e6,color=GRAY)
    bb_ai.plot(a.year,positive.to_numpy()/1e3,color=ORANGE)
    bb.set_ylabel('Papers (millions)');bb_ai.set_ylabel('Papers (thousands)')
    bb.text(.5,1.01,GROUP_LABELS['known'],transform=bb.transAxes,va='bottom',ha='center',fontsize=5.5,color=GRAY)
    bb_ai.text(.5,1.01,GROUP_LABELS['ai'],transform=bb_ai.transAxes,va='bottom',ha='center',fontsize=5.5,color=ORANGE)
    bb.tick_params(labelbottom=False);bb_ai.set_xlabel('Year')
    bb.set_ylim(bottom=0);bb_ai.set_ylim(bottom=0);lab(bb,'b')
    cc.plot(a.year,a.AIProb_mean,color=PURPLE);cc.set_xlabel('Year');cc.set_ylabel('Mean AI score');lab(cc,'c')
    examples=representative_sources(c)
    source_data=read(c,'source_year_groups.csv')
    source_data=source_data[source_data.source_id.isin(examples.source_id)]
    denom=source_data.groupby(['year','source_id']).n.sum()
    numerator=source_data[source_data.ai_group.eq(1)].set_index(['year','source_id']).n.reindex(denom.index,fill_value=0)
    share=(100*numerator/denom).rename('share').reset_index()
    heatmap=share[share.year.ge(c['trend_start'])].pivot(index='source_id',columns='year',values='share').reindex(examples.source_id)
    im=heat(dd,heatmap);dd.set_xlabel('Year');dd.set_ylabel('Example publishing source')
    dd.set_yticks(range(len(examples)),examples.source_label,fontsize=5)
    if len(heatmap.columns):dd.set_xticks([0,len(heatmap.columns)-1],[heatmap.columns.min(),heatmap.columns.max()])
    if im is not None:fig.colorbar(im,ax=dd,orientation='horizontal',fraction=.045,pad=.08,label='AI-positive share (%)')
    lab(dd,'d')
    h=read(c,'histograms.csv');h=h[h.metric.eq('AIProb')].groupby('bin').agg({'count':'sum','left':'first','right':'first'})
    ee.bar((h.left+h.right)/2,h['count'],width=h.right-h.left,color=PURPLE)
    ee.set_yscale('log');ee.set_xlabel('AI score');ee.set_ylabel('Scored papers (log scale)');lab(ee,'e')
    ff.scatter(recent.known,100*recent.share,s=8,color=TEAL,alpha=.65)
    ff.set_xscale('log');ff.set_xlabel('Scored country credit (log scale)');ff.set_ylabel('AI-positive share (%)');lab(ff,'f')
    trend=raw.groupby('year')[['known','ai_gt02','ai','ai_gt08']].sum()
    trend=trend[trend.index>=c['trend_start']]
    for key,col,label in [('ai_gt02',TEAL,'>0.2'),('ai',ORANGE,'>0.5'),('ai_gt08',PURPLE,'>0.8')]:
        gg.plot(trend.index,100*safe_ratio(trend[key],trend.known),color=col,label=label)
    gg.set_xlabel('Year');gg.set_ylabel('Share of scored credit (%)');gg.legend(frameon=False,ncol=3);lab(gg,'g')
    save(fig,c,'Fig1_AI_Extent','The population includes all core papers with an observed AI score and an eligible publication year. a,f,g use unique-paper country credit. b displays All scored and AI-positive counts in two aligned plots with separate linear scales (millions and thousands). c shows the unique-paper mean AI score. d shows fixed example publishing sources, chosen by 2020-2024 scored output among sources meeting the declared two-group support threshold; the source_catalog table records their IDs, labels and sample sizes. Example selection does not restrict any population summary. e shows the score distribution. g uses thresholds >0.2, >0.5 and >0.8 with the same scored denominator.')


def figure2(c):
    z=read(c,'country_source_rca_period.csv');latest=z[z.period.eq(2020)]
    if latest.empty:return
    unique=pd.read_parquet(dirs(c)['data']/'country_masses.parquet')
    top=unique[unique.year.between(2020,c['end_year'])].groupby('country').known.sum().nlargest(25).index
    examples=representative_sources(c);sources=examples.source_id.tolist();fig=start(c,height=190)
    gs=fig.add_gridspec(4,1,height_ratios=[1.65,1,.80,1])
    top_grid=gs[0].subgridspec(1,2);middle_grid=gs[1].subgridspec(1,2);bottom_grid=gs[3].subgridspec(1,2)
    a=fig.add_subplot(top_grid[0]);b=fig.add_subplot(top_grid[1]);cc=fig.add_subplot(middle_grid[0]);dd=fig.add_subplot(middle_grid[1])
    e=fig.add_subplot(gs[2]);f=fig.add_subplot(bottom_grid[0]);g=fig.add_subplot(bottom_grid[1])
    for ax,key,letter in [(a,'rsca_ai','a'),(b,'rsca_nonai','b')]:
        p=latest.pivot(index='country',columns='source_id',values=key).reindex(index=top,columns=sources)
        im=heat(ax,p,cmap='RdBu_r',vmin=-1,vmax=1,labels=True);ax.set_xlabel('Example publishing source')
        ax.set_xticks(range(len(sources)),examples.source_label,rotation=45,ha='right',fontsize=5)
        ax.text(.5,1.02,GROUP_LABELS['ai' if key=='rsca_ai' else 'nonai'],ha='center',va='bottom',transform=ax.transAxes,fontsize=6)
        lab(ax,letter)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),
                    ax=[a,b],orientation='horizontal',fraction=.05,pad=.05,
                    shrink=.65,aspect=45)
    cb.set_ticks([-1,0,1]);cb.set_label('RSCA',fontsize=5.4,labelpad=0);cb.ax.tick_params(labelsize=5,pad=1)
    v=latest[latest.known.ge(c['min_cell_period_mass'])]
    cc.scatter(v.rsca_nonai,v.rsca_ai,s=3,alpha=.3,color=PURPLE,rasterized=True);cc.plot([-1,1],[-1,1],':',lw=.6,color=GRAY);cc.set_xlabel('RSCA: Other-scored');cc.set_ylabel('RSCA: AI-positive');lab(cc,'c')
    grid=np.linspace(-1,1,150)
    for j,pd0 in enumerate([2005,2010,2015,2020]):
        vals=z.loc[z.period.eq(pd0)&z.known.ge(c['min_cell_period_mass']),'rsca_ai'].dropna()
        den=weighted_density(vals,grid)
        if den.max():den=den/den.max()*.8
        dd.fill_between(grid,j,j+den,color=[BLUE,TEAL,PURPLE,ORANGE][j],alpha=.7)
    dd.set_yticks(range(4),['2005-09','2010-14','2015-19','2020-24']);dd.set_xlabel('AI-positive RSCA');lab(dd,'d')
    states=read(c,'specialization_states.csv')
    alluvial(e,states[states.source_id.isin(sources)],[2005,2010,2015,2020]);lab(e,'e')
    q=v.replace([np.inf,-np.inf],np.nan).dropna(subset=['ai_lq','rca_known'])
    f.scatter(np.log1p(q.rca_known),np.log1p(q.ai_lq),s=3,color=TEAL,alpha=.3,rasterized=True);f.set_xlabel('log(1 + RCA): All scored');f.set_ylabel('log(1 + LQ)\nAI-positive');lab(f,'f')
    q=v.dropna(subset=['rca_ai','rca_fixed_ai'])
    g.scatter(np.log1p(q.rca_ai),np.log1p(q.rca_fixed_ai),s=3,color=ORANGE,alpha=.3,rasterized=True);g.set_xlabel('log(1 + RCA): Period benchmark');g.set_ylabel('log(1 + RCA)\nFixed benchmark');lab(g,'g')
    save(fig,c,'Fig2_AI_Specialization','RCA is based on unique-paper country credit over all publishing sources. AI-positive means AIProb>0.5 and Other-scored means AIProb<=0.5. RSCA=(RCA-1)/(RCA+1), retaining observed AI zeros when denominators exist. a,b show the fixed example sources and the 25 largest countries by All scored output in the same order. c,d,f,g use all supported country-source cells. e tracks the union of country-example-source units across five-year windows; absent or low-support units are shown separately. f is the local country-source AI rate divided by the global rate for the same source. g compares each period benchmark with global AI-positive source shares in 2010-2014. These are publishing-venue portfolios, not content classifications.')


def fixed_support_panel(ax,fixed,c):
    """Display countries supported in each declared portfolio window."""
    n=0 if fixed.empty else int(fixed.n_countries.min())
    width=c.get('portfolio_window_years',5)
    if n<2:
        message=(f'No supported between-country panel\n'
                 f'{n} eligible countries, {c["fixed_support_start"]}-{c["end_year"]}\n'
                 f'In every {width}-year window: scored >= {c["min_country_period_mass"]:g},\n'
                 f'other scored >= {c["min_country_period_mass"]:g}, '
                 f'AI >= {c.get("min_ai_period_mass",3):g}')
        ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,
                color=GRAY,fontsize=6,linespacing=1.5)
        ax.set_axis_off()
        return n
    for metric,col,label in [('ai',ORANGE,'AI-positive'),('nonai',BLUE,'Other-scored'),
                              ('known',GRAY,'All scored')]:
        z=fixed[fixed.metric.eq(metric)]
        ax.plot(z.year,z.effective_between,color=col,label=label,
                marker='o' if z.year.nunique()==1 else None,markersize=3)
    if fixed.year.nunique()==1:
        year=int(fixed.year.iloc[0]);ax.set_xlim(year-1,year+1);ax.set_xticks([year])
    ax.legend(frameon=False);period_axis(ax,fixed.year,c)
    ax.set_ylabel('Effective between-country diversity')
    ax.text(.02,.97,f'{n} fixed countries',ha='left',va='top',
            transform=ax.transAxes,fontsize=5.2,color=GRAY)
    return n

def pair_distance_panel(ax,js,years=(2000,2010,2020,2024),period_labels=None):
    """Show observed pairs explicitly when a density cannot be estimated."""
    grid=np.linspace(0,np.log(2),150);support=[];labels=[]
    for j,yr in enumerate(years):
        vals=(js.loc[js.year.eq(yr)&js.metric.eq('ai'),'jsd'].to_numpy(float)
              if not js.empty else np.array([],dtype=float))
        vals=vals[np.isfinite(vals)];n=len(vals)
        label=period_labels.get(yr,str(yr)) if period_labels else str(yr)
        labels.append(f'{label} (n={n:,})')
        col=[BLUE,TEAL,PURPLE,ORANGE][j%4]
        if not n:
            ax.text(.5,j+.08,'No eligible pairs',ha='center',va='bottom',
                    transform=ax.get_yaxis_transform(),fontsize=5.3,color=GRAY)
            display='unavailable'
        else:
            den=None
            if n>=4 and np.ptp(vals)>=1e-10:
                try:den=weighted_density(vals,grid)
                except (ValueError,np.linalg.LinAlgError):pass
            if den is not None and np.isfinite(den).all() and den.max()>0:
                ax.fill_between(grid,j,j+den/den.max()*.8,color=col,alpha=.7)
                display='density'
            else:
                ax.plot(vals,np.full(n,j+.07),linestyle='none',marker='|',
                        markersize=5,color=col,clip_on=False,zorder=3)
                display='observations'
        support.append(dict(year=yr,n_pairs=n,display=display))
    ax.set_yticks(range(len(years)),labels)
    ax.set_ylim(-.18,len(years)-.02);ax.set_xlim(0,np.log(2))
    ax.set_xlabel('Jensen-Shannon divergence (nats)')
    return support

def period_label(year,c):
    width=c.get('portfolio_window_years',5)
    return str(year) if width==1 else f'{year}-{str(min(year+width-1,c["end_year"]))[-2:]}'

def period_axis(ax,years,c):
    years=sorted(set(int(y) for y in years))
    shown=years[::max(1,int(np.ceil(len(years)/5)))]
    ax.set_xticks(shown,[period_label(y,c) for y in shown])
    ax.set_xlabel('Publication window' if c.get('portfolio_window_years',5)>1 else 'Year')

def figure3(c):
    export_publication_tables(c)
    dv=read_table(c,'diversity_period_decomposition.csv')
    fixed=read_table(c,'diversity_fixed_support_period.csv')
    js=read_table(c,'country_pair_jsd_period.csv')
    fig=start(c,height=170)
    gs=fig.add_gridspec(3,2)
    ax=[fig.add_subplot(gs[j,k]) for j in range(3) for k in range(2)]
    a,b,cc,dd,e,f=ax
    z=dv[dv.year.ge(c['trend_start'])&dv.metric.eq('ai')&dv.weighting.eq('mass')]
    for key,col,label in [('global_h',PURPLE,'Global'),('within_h',BLUE,'Within countries'),('between_h',ORANGE,'Between countries')]:a.plot(z.year,z[key],color=col,label=label)
    period_axis(a,z.year,c);a.set_ylabel('Entropy (nats)');a.legend(frameon=False);lab(a,'a')
    fixed_support_panel(b,fixed,c)
    lab(b,'b')
    pair_years=([2000,2010,2015,2020] if c.get('portfolio_window_years',5)==5
                else [2000,2010,2020,2024])
    pair_distance_panel(cc,js,pair_years,{y:period_label(y,c) for y in pair_years})
    lab(cc,'c')
    q=read(c,'ai_change_accounting.csv')
    if len(q):
      q=q[q.measure.eq('binary_share')];cum=0
      for j,row in enumerate(q.itertuples()):
        value=100*row.contribution
        if row.component=='total':bottom=0;color=PURPLE
        else:bottom=cum;color=TEAL if value>=0 else ORANGE;cum+=value
        dd.bar(j,value,bottom=bottom,color=color,width=.7)
        dd.annotate(f'{value:.3g}' if value else '0',(j,bottom+value),
                    xytext=(0,3 if value>=0 else -3),textcoords='offset points',
                    ha='center',va='bottom' if value>=0 else 'top',fontsize=5.2)
      dd.use_sticky_edges=False;dd.margins(y=.16)
      dd.set_xticks(range(len(q)),q.component,rotation=30,ha='right');dd.axhline(0,color=GRAY,lw=.5);dd.set_ylabel('Percentage-point change')
    else:no_data(dd)
    lab(dd,'d')
    z=read(c,f'TableS13_country_standardization_{c["end_year"]}.csv')
    if len(z):
      e.scatter(z.raw_ai_positive_pct,z.standardized_ai_positive_pct,s=8,color=TEAL,alpha=.65,rasterized=True)
      limit=max(float(z.raw_ai_positive_pct.max()),float(z.standardized_ai_positive_pct.max()),.01)*1.05
      e.plot([0,limit],[0,limit],':',color=GRAY,lw=.65,zorder=0)
      e.text(.03,.97,f'{c["end_year"]}; {len(z)} countries',ha='left',va='top',
             transform=e.transAxes,fontsize=5.3,color=GRAY)
      e.set_xlabel('Raw AI-positive source share (%)');e.set_ylabel('Source-adjusted share (%)')
    else:no_data(e,'No country-source panel meets common-support rules')
    lab(e,'e')
    for w,col in [('mass',ORANGE),('equal_country',BLUE)]:
      z=dv[dv.year.ge(c['trend_start'])&dv.metric.eq('ai')&dv.weighting.eq(w)];f.plot(z.year,z.between_h,color=col,label='Paper credit' if w=='mass' else 'Equal countries')
    period_axis(f,z.year,c);f.set_ylabel('Between-country entropy (nats)');f.legend(frameon=False);lab(f,'f')
    save(fig,c,'Fig3_Diversity_Accounting',"All source portfolios use each country-attributed paper once at its publishing source. a,b,c,f pool source credit over non-overlapping publication windows before computing entropy or distance. a partitions global source entropy into within-country and between-country components. b fixes countries supported in every configured window and their 2010-2014 All scored weights. c compares supported AI-positive country-source portfolios; empty and sparse pair samples are shown explicitly. d exactly decomposes the AI-share change in country-attributed papers between the configured baseline and comparison periods using country by publishing-source cells. e is the 2024 country cross-section on common-source support; it uses the same full-precision rows as Table S13. Sources need the configured scored credit in every standardization window, positive baseline benchmark weight, and an observed target-year denominator. Raw and adjusted rates use identical retained source support; Table S13 also reports each country's unrestricted scored-paper rate. f compares paper-credit and equal-country weights. Source portfolios describe publishing venues; they are not research-content diversity.")


def source_adjusted_trend(ax,contrasts,metric,color,ylabel,legend=False):
    """Join consecutive years only when the supported publishing-source set is unchanged."""
    z=contrasts[contrasts.metric.eq(metric)].sort_values('year')
    for _,segment in z.groupby('support_segment'):
        ax.plot(segment.year,segment.source_adjusted_difference,color=color,lw=1)
    supported=z.source_adjusted_difference.notna()
    ax.scatter(z.loc[supported,'year'],z.loc[supported,'source_adjusted_difference'],
               s=9,edgecolors=color,facecolors=color,linewidths=.65,zorder=3)
    if not supported.any():no_data(ax)
    ax.axhline(0,color=GRAY,lw=.6,ls=':')
    ax.set_xlabel('Year');ax.set_ylabel(ylabel)
    if legend:
        ax.plot([],[],marker='o',color=color,markersize=3,label='Common-source adjustment')
        ax.legend(frameon=False,fontsize=5,loc='best')
    return z


def figure4(c):
    sample=pd.read_parquet(dirs(c)['data']/'plot_sample.parquet');con=read(c,'profile_source_adjusted_contrasts.csv')
    surf=read(c,'response_surface.csv');source_contrasts=read(c,'profile_source_year_contrasts.csv');gg=read(c,'annual_groups.csv')
    fig=start(c,height=205)
    gs=fig.add_gridspec(4,1,height_ratios=[.85,.90,.85,1.25])
    top_grid=gs[0].subgridspec(1,2);contrast_grid=gs[1].subgridspec(1,2)
    middle_grid=gs[2].subgridspec(1,2,width_ratios=[1.8,1]);bottom_grid=gs[3].subgridspec(1,2)
    a=fig.add_subplot(top_grid[0]);b=fig.add_subplot(top_grid[1])
    cc=fig.add_subplot(contrast_grid[0]);cc_sem=fig.add_subplot(contrast_grid[1])
    dd=fig.add_subplot(middle_grid[0]);e=fig.add_subplot(middle_grid[1])
    cc.set_gid('Fig4c_Simpson');cc_sem.set_gid('Fig4c_Semantic');dd.set_gid('Fig4d_Sources')
    f=fig.add_subplot(bottom_grid[0]);g=fig.add_subplot(bottom_grid[1],projection='3d')
    rain(a,sample,'ref_field_simpson','Reference-field Simpson diversity');lab(a,'a')
    rain(b,sample,'emb_ref_pair_mean_distance','Mean reference-pair semantic distance');b.set_yticklabels([]);lab(b,'b')
    if len(con):
        source_adjusted_trend(cc,con,'ref_field_simpson',ORANGE,
            'Source-adjusted difference\n(Simpson units)',legend=True)
        source_adjusted_trend(cc_sem,con,'emb_ref_pair_mean_distance',BLUE,
            'Source-adjusted difference\n(cosine-distance units)')
    else:no_data(cc);no_data(cc_sem)
    lab(cc,'c(i)');lab(cc_sem,'c(ii)')
    examples=representative_sources(c)
    q=source_contrasts[source_contrasts.metric.eq('ref_field_simpson')]
    years=list(range(c['trend_start'],c['end_year']+1))
    p=(q.pivot(index='source_id',columns='year',values='difference') if len(q)
        else pd.DataFrame()).reindex(index=examples.source_id,columns=years)
    extent=float(np.nanmax(np.abs(p.to_numpy(float)))) if p.notna().any().any() else .2
    extent=max(.05,np.ceil(extent/.05)*.05)
    im=heat(dd,p,cmap='RdBu_r',vmin=-extent,vmax=extent)
    dd.set_xticks([0,len(years)//2,len(years)-1],[years[0],years[len(years)//2],years[-1]])
    dd.set_yticks(range(len(examples)),examples.source_label,fontsize=5.6)
    dd.set_xlabel('Year');dd.set_ylabel('Example publishing source')
    if im is not None:
        cb=fig.colorbar(im,ax=dd,orientation='horizontal',fraction=.055,pad=.06,aspect=35)
        cb.set_ticks([-extent,0,extent]);cb.set_label('AI-positive minus Other-scored (Simpson units)',fontsize=5.1)
    lab(dd,'d')
    for group in [0,1]:
      z=gg[gg.ai_group.eq(group)&gg.year.ge(c['trend_start'])]
      e.plot(z.year,100*z.emb_ref_pair_mean_distance_coverage,color=GROUPS[group][1],label=GROUPS[group][0])
    e.set_ylim(0,100);e.set_xlabel('Year');e.set_ylabel('Semantic-metric coverage (%)');e.legend(frameon=False);lab(e,'e')
    if len(surf):
      p=surf.pivot(index='ref_bin',columns='ai_bin',values='mean').reindex(index=range(8),columns=range(10))
      support=surf.pivot(index='ref_bin',columns='ai_bin',values='supported').reindex(index=range(8),columns=range(10)).astype('boolean').fillna(False).astype(bool)
      p=p.where(support);vmin=float(np.nanmin(p)) if p.notna().any().any() else 0;vmax=float(np.nanmax(p)) if p.notna().any().any() else 1
      im=heat(f,p,vmin=vmin,vmax=vmax);f.set_xticks([0,4,9],['0-.1','.4-.5','.9-1']);f.set_yticks(range(8),['0-1','2-4','5-9','10-19','20-39','40-79','80-159','160-319'])
      f.set_xlabel('AI score bin');f.set_ylabel('Core reference count')
      if im is not None:
        fig.colorbar(im,ax=f,orientation='horizontal',fraction=.045,pad=.06,
                     aspect=35,label='Mean field diversity')
      xx,yy=np.meshgrid(np.arange(10),np.arange(8));zz=p.to_numpy(float)
      if np.isfinite(zz).sum()>=4:
        g.plot_surface(xx,yy,zz,cmap='viridis',vmin=vmin,vmax=vmax,linewidth=.15,edgecolor='#FFFFFF',antialiased=True)
      g.set_xticks([0,4,9],['0-.1','.4-.5','.9-1']);g.set_yticks([0,3,7],['0-1','10-19','160-319'])
      g.set_xlabel('AI score bin',labelpad=1);g.set_ylabel('Reference count',labelpad=2);g.set_zlabel('Mean diversity',labelpad=1);g.tick_params(labelsize=5,pad=0);g.view_init(26,-130)
      g.set_box_aspect((1,1,.70),zoom=.90)
    else:no_data(f);no_data(g)
    lab(f,'f');lab(g,'g')
    save(fig,c,'Fig4_Knowledge_Profiles','AI-positive means AIProb>0.5; Other-scored means AIProb<=0.5. The supplied reference-field diversity, semantic distances and historical-centroid indicators retain their local definitions and values. No paper is selected using a six-domain label. a,b display deterministic samples uniform within score strata for 2015-2024. c(i),c(ii) show source-adjusted mean differences in the original Simpson and cosine-distance units. Each metric-year comparison uses all publishing sources with the configured minimum measurable papers in both groups, weighted by pooled measurable paper counts. Lines join years only while the supported source set is unchanged. Table S11 reports source counts, identities and retained paper fractions. d shows within-source reference-field Simpson differences for the fixed example sources, with unsupported cells grey. e uses unique-paper denominators for semantic-metric coverage. f,g retain the precomputed reference-field Simpson outcome and share score/reference-count bins, supported means and color limits; each cell requires the configured minimum measurable unique papers. References >=320 remain in other summaries.')


def figure5(c):
    export_publication_tables(c)
    cohorts=read(c,'reference_cohorts.csv');lor=read(c,'country_lorenz.csv')
    knowledge=pd.DataFrame()
    summary=read(c,f'Table7_country_concentration_{c["end_year"]}.csv')
    raw=pd.read_parquet(dirs(c)['data']/'country_masses.parquet')
    fig=start(c,height=165)
    gs=fig.add_gridspec(3,1,height_ratios=[1.15,1,1])
    middle=gs[1].subgridspec(1,3);bottom=gs[2].subgridspec(1,3)
    a=fig.add_subplot(gs[0]);b=fig.add_subplot(middle[0]);cc=fig.add_subplot(middle[1]);dd=fig.add_subplot(middle[2])
    e=fig.add_subplot(bottom[0]);f=fig.add_subplot(bottom[1]);g=fig.add_subplot(bottom[2])
    if len(cohorts):
        z=cohorts[cohorts.citing_year.between(c['trend_start'],c['end_year'])].copy()
        z['cohort']=(z.cited_year//5)*5
        z['cohort']=np.where(z.cited_year<1900,1895,z.cohort)
        matrix=z.groupby(['citing_year','cohort']).edges.sum().unstack(fill_value=0)
        matrix=matrix.reindex(index=range(c['trend_start'],c['end_year']+1),columns=[1895]+list(range(1900,c['end_year']+1,5)),fill_value=0)
        matrix=matrix.div(matrix.sum(1).replace(0,np.nan),axis=0)
        matrix=matrix.mask(np.asarray(matrix.columns)[None,:]>np.asarray(matrix.index)[:,None])
        values=matrix.to_numpy();finite=values[np.isfinite(values)]
        vmax=max(.05,float(np.quantile(finite,.99))) if len(finite) else 1
        im=heat(a,matrix,cmap='magma',vmin=0,vmax=vmax)
        xt=np.linspace(0,len(matrix.columns)-1,min(9,len(matrix.columns)),dtype=int)
        yt=np.linspace(0,len(matrix)-1,min(5,len(matrix)),dtype=int)
        a.set_xticks(xt,['<1900' if matrix.columns[j]==1895 else str(matrix.columns[j]) for j in xt]);a.set_yticks(yt,matrix.index[yt])
        a.set_xlabel('Cited publication cohort (five-year start)');a.set_ylabel('Citing publication year')
        if im is not None:fig.colorbar(im,ax=a,orientation='horizontal',fraction=.045,pad=.045,shrink=.65,label='Share of references from selected papers')
        recent=z[z.citing_year.between(*c['comparison_years'])]
        for group in [0,1]:
            q=recent[recent.ai_group.eq(group)].groupby('reference_age').edges.sum().sort_index()
            if q.sum():b.plot(q.index,q.cumsum()/q.sum(),color=GROUPS[group][1],label=GROUPS[group][0])
        b.set_xlim(0,80);b.set_ylim(0,1)
        b.set_xlabel('Reference age (years)');b.set_ylabel('Cumulative fraction');b.legend(frameon=False)
        q=z.groupby(['citing_year','reference_ai_group']).edges.sum().unstack(fill_value=0)
        denom=q.reindex(columns=[0,1],fill_value=0).sum(1)
        f.plot(q.index,100*safe_ratio(q.get(1,pd.Series(0,index=q.index)),denom),color=ORANGE)
        f.set_xlabel('Citing year');f.set_ylabel('AI share in scored references (%)')
    else:no_data(a);no_data(b);no_data(f)
    lab(a,'a');lab(b,'b');lab(f,'f')
    for metric,col in [('known',GRAY),('ai',ORANGE),('nonai',BLUE)]:
        q=lor[lor.year.eq(c['end_year'])&lor.metric.eq(metric)]
        if len(q):cc.plot(q.population_fraction,q.cumulative_mass,color=col,label=GROUP_LABELS[metric])
    cc.plot([0,1],[0,1],':',color=GRAY,lw=.6)
    cc.set_xlabel('Fraction of eligible countries');cc.set_ylabel('Cumulative output share');cc.legend(frameon=False);lab(cc,'c')
    bars=summary.set_index('metric').loc[['known','ai','nonai']]
    dd.bar(range(3),bars.top_five_pct,color=[GRAY,ORANGE,BLUE])
    dd.set_xticks(range(3),[GROUP_LABELS[k] for k in ['known','ai','nonai']],rotation=20,ha='right');dd.set_ylabel('Top-five country share (%)');lab(dd,'d')
    country=raw[raw.year.between(*c['comparison_years'])].groupby('country')[['known','ai']].sum()
    country['ai_share']=safe_ratio(country.ai,country.known)
    local=raw[raw.year.between(*c['comparison_years'])].groupby('country')[['mean_ref_age_numer','mean_ref_age_denom']].sum()
    local['reference_age']=safe_ratio(local.mean_ref_age_numer,local.mean_ref_age_denom)
    qq=country.join(local,how='inner').dropna(subset=['reference_age'])
    e.scatter(100*qq.ai_share,qq.reference_age,s=9,color=TEAL,alpha=.65)
    e.set_xlabel('AI-positive share (%)');e.set_ylabel('Mean reference age (years)')
    lab(e,'e')
    countries=raw[raw.year.eq(c['end_year'])].groupby('country')[['known','ai']].sum()
    top=countries.nlargest(12,'known').copy();top['share']=100*safe_ratio(top.ai,top.known)
    g.barh(range(len(top)),top.share,color=TEAL,alpha=.7)
    global_share=100*countries.ai.sum()/countries.known.sum()
    g.axvline(global_share,color=GRAY,ls=':',lw=.7)
    g.set_yticks(range(len(top)),top.index);g.invert_yaxis();g.set_xlabel('AI-positive share (%)');lab(g,'g')
    save(fig,c,'Fig5_Cohorts_Concentration','All scored core papers with eligible publication years enter the population. AI-positive means AIProb>0.5 and Other-scored means AIProb<=0.5. Existing local paper indicators are joined without recalculation. a,b aggregate supplied citation edges from scored citing papers, retaining the local reference universe. b is reference-event weighted; its 80-year limit is display only. c,d use unique-paper country credit in 2024. d reads the full-precision top-five percentages from Table 7. e summarizes supplied mean reference age. f is AI-positive share among scored references. g shows AI prevalence in the largest All scored output countries in 2024.')


def make_figures(c):
    require_scope(c)
    style()
    for func in [figure1,figure2,figure3,figure4,figure5]:
        print(func.__name__,flush=True);func(c)
    files=sorted(p.name for p in dirs(c)['figures'].glob('*'))
    json_write(files,dirs(c)['status']/'figure_files.json');return files
