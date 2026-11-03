import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from atlas.synthetic import make_fixture,fixture_config
from atlas.prepare import build_master,METRICS
from atlas.describe import build_descriptions
from atlas.portfolios import country_masses,build_portfolios
from atlas.knowledge import profile_comparisons,reference_cohorts
from atlas.sources import representative_sources
from atlas.figures import figure1,figure2,figure3,figure4,figure5,style

class EndToEndChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        make_fixture(cls.root/'input',n=3000)
        cls.c=fixture_config(cls.root/'input',cls.root/'results');cls.c['batch_rows']=500
        for fn in [build_master,build_descriptions,country_masses,build_portfolios,profile_comparisons,reference_cohorts]:fn(cls.c)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_selected_population_without_a_label_relation(self):
        z=pd.read_parquet(self.root/'results/data/papers.parquet')
        core=pd.read_parquet(self.root/'input/core/papers.parquet')
        scores=pd.read_parquet(self.root/'input/extended/AI_papers.parquet')
        expected=core.merge(scores,on='paper_id')
        expected=expected[expected.AIProb.notna()&expected.year.between(1980,2024)]
        self.assertEqual(set(z.paper_id),set(expected.paper_id))
        self.assertTrue(z.AIProb.notna().all())
        self.assertEqual(str(z.paper_id.dtype),'uint64')
        self.assertNotIn('primary_field_id',z)
        self.assertFalse((self.root/'input/prepared').exists())
        catalog=pd.read_csv(self.root/'results/tables/source_catalog.csv')
        self.assertEqual(catalog.all_scored.sum(),len(z))
        self.assertLessEqual(int(catalog.selected.sum()),6)

    def test_source_partitions_conserve_unique_paper_counts(self):
        m=pd.read_parquet(self.root/'results/data/papers.parquet')
        annual=pd.read_csv(self.root/'results/tables/annual_all.csv')
        by_source=pd.read_csv(self.root/'results/tables/source_year_groups.csv')
        self.assertEqual(int(annual.n.sum()),len(m))
        self.assertEqual(int(by_source.n.sum()),len(m))
        m['ai_group']=m.AIProb.gt(.5).astype(int)
        expected=m.groupby(['year','source_id','ai_group']).size().sort_index()
        actual=by_source.set_index(['year','source_id','ai_group']).n.sort_index()
        np.testing.assert_array_equal(expected,actual)
        population=m.AIProb.gt(.5).sum()
        self.assertEqual(by_source.loc[by_source.ai_group.eq(1),'n'].sum(),population)

    def test_score_threshold_precision(self):
        z=pd.read_parquet(self.root/'results/data/papers.parquet').set_index('paper_id')
        self.assertEqual(z.loc[np.uint64(2**53+12),'AIProb'],.5)
        self.assertGreater(z.loc[np.uint64(2**53+15),'AIProb'],.5)
        self.assertLess(z.loc[np.uint64(2**53+16),'AIProb'],.5)
    def test_local_metrics_are_unchanged(self):
        z=pd.read_parquet(self.root/'results/data/papers.parquet').set_index('paper_id')
        core=pd.read_parquet(self.root/'input/core/papers.parquet').set_index('paper_id').reindex(z.index)
        for col in ['n_references','mean_ref_age','c5_norm','c5_pct','top10_credit']:np.testing.assert_array_equal(z[col],core[col])
        for key,columns in METRICS.items():
            origin=pd.read_parquet(self.root/'input'/self.c['paths'][key]).set_index('paper_id').reindex(z.index)
            for col in columns:np.testing.assert_array_equal(z[col],origin[col])
        self.assertTrue(z.ref_field_shannon.gt(np.log(6)).any())
    def test_country_mass_conservation(self):
        x=pd.read_parquet(self.root/'results/data/country_source_masses.parquet')
        np.testing.assert_allclose(x.ai+x.nonai,x.known,atol=1e-10)
        self.assertFalse('mass' in x);self.assertTrue((x.score<=x.known+1e-10).all())
        members=pd.read_parquet(self.root/'results/data/country_members_scoped.parquet')
        np.testing.assert_allclose(members.groupby('paper_id').weight.sum(),1,atol=1e-12)
        unique=pd.read_parquet(self.root/'results/data/country_masses.parquet')
        self.assertAlmostEqual(unique.known.sum(),members.paper_id.nunique())
        np.testing.assert_allclose(x.groupby(['year','country']).known.sum(),unique.set_index(['year','country']).known.sort_index())
        self.assertAlmostEqual(x.known.sum(),unique.known.sum())
    def test_concentration_uses_unique_papers(self):
        z=pd.read_parquet(self.root/'results/data/country_masses.parquet')
        source=z[z.year.eq(2024)].set_index('country')
        table=pd.read_csv(self.root/'results/tables/Table7_country_concentration_2024.csv').set_index('metric')
        for metric in ['known','ai','nonai','score']:
            p=source[metric]/source[metric].sum()
            self.assertAlmostEqual(table.loc[metric,'hhi'],float(p@p))
        q=pd.read_csv(self.root/'results/tables/ai_change_accounting.csv')
        a=z[z.year.between(*self.c['baseline_years'])][['known','ai']].sum()
        b=z[z.year.between(*self.c['comparison_years'])][['known','ai']].sum()
        actual=q[q.measure.eq('binary_share')&q.component.eq('total')].contribution.iloc[0]
        self.assertAlmostEqual(actual,b.ai/b.known-a.ai/a.known)
    def test_r_sample_maturity(self):
        z=pd.read_parquet(self.root/'results/data/r_papers.parquet')
        self.assertTrue(z.AIProb.notna().all());self.assertTrue(z.source_id.notna().all())
        self.assertFalse(z.paper_id.duplicated().any())
        self.assertTrue(z.loc[z.year>2019,'c5_norm'].isna().all())
    def test_histogram_denominators(self):
        h=pd.read_csv(self.root/'results/tables/histograms.csv');a=pd.read_csv(self.root/'results/tables/annual_groups.csv')
        self.assertTrue(a.ai_group.isin([0,1]).all())
        for metric in ['AIProb','ref_field_simpson','team_size']:
            hh=h[h.metric==metric].groupby(['year','ai_group'])[['count','under','over']].sum().sum(axis=1)
            aa=a[a.year>=self.c['trend_start']].set_index(['year','ai_group'])[metric+'_n']
            np.testing.assert_array_equal(hh.reindex(aa.index),aa)
    def test_source_entropy_and_cohort_scope(self):
        h=pd.read_csv(self.root/'results/tables/diversity_decomposition.csv')
        np.testing.assert_allclose(h.global_h,h.within_h+h.between_h,atol=2e-12)
        self.assertTrue((h.global_h.dropna()<=np.log(12)+1e-12).all())
        x=pd.read_csv(self.root/'results/tables/reference_cohorts.csv')
        self.assertTrue(x.ai_group.isin([0,1]).all());self.assertTrue((x.reference_age>=0).all())
    def test_fig5d_matches_table7(self):
        seen={}
        with patch('atlas.figures.save',side_effect=lambda fig,c,name,caption:seen.setdefault(name,fig)):figure5(self.c)
        fig=seen['Fig5_Cohorts_Concentration'];heights=[p.get_height() for p in fig.axes[3].patches]
        table=pd.read_csv(self.root/'results/tables/Table7_country_concentration_2024.csv').set_index('metric')
        np.testing.assert_allclose(heights,table.loc[['known','ai','nonai'],'top_five_pct'],rtol=0,atol=1e-12);plt.close(fig)
    def test_fig3e_matches_table_s13(self):
        seen={}
        with patch('atlas.figures.save',side_effect=lambda fig,c,name,caption:seen.setdefault(name,fig)):figure3(self.c)
        fig=seen['Fig3_Diversity_Accounting'];table=pd.read_csv(self.root/'results/tables/TableS13_country_standardization_2024.csv')
        self.assertGreater(len(table),0);self.assertTrue(table.year.eq(2024).all())
        self.assertEqual(len(table),table.country.nunique())
        np.testing.assert_allclose(fig.axes[4].collections[0].get_offsets(),table[['raw_ai_positive_pct','standardized_ai_positive_pct']],rtol=0,atol=1e-12);plt.close(fig)
    def test_figure1_separate_count_scales(self):
        seen={};style()
        with patch('atlas.figures.save',side_effect=lambda fig,c,name,caption:seen.setdefault(name,fig)):figure1(self.c)
        fig=seen['Fig1_AI_Extent'];axes={a.get_gid():a for a in fig.axes if a.get_gid()}
        all_ax=axes['Fig1b_AllScored'];ai_ax=axes['Fig1b_AIpositive']
        self.assertIsNot(all_ax,ai_ax)
        self.assertIn('millions',all_ax.get_ylabel());self.assertIn('thousands',ai_ax.get_ylabel())
        source=pd.read_csv(self.root/'results/tables/annual_groups.csv')
        expected=source[source.ai_group.eq(1)].set_index('year').n.reindex(ai_ax.lines[0].get_xdata(),fill_value=0)
        np.testing.assert_allclose(ai_ax.lines[0].get_ydata(),expected/1000)
        plt.close(fig)
    def test_figure4_retains_precomputed_units_and_uses_source_rows(self):
        seen={};style()
        with patch('atlas.figures.save',side_effect=lambda fig,c,name,caption:seen.setdefault(name,fig)):figure4(self.c)
        fig=seen['Fig4_Knowledge_Profiles'];axes={a.get_gid():a for a in fig.axes if a.get_gid()}
        self.assertIn('Simpson units',axes['Fig4c_Simpson'].get_ylabel())
        self.assertIn('cosine-distance units',axes['Fig4c_Semantic'].get_ylabel())
        self.assertEqual([x.get_text() for x in axes['Fig4d_Sources'].get_yticklabels()],representative_sources(self.c).source_label.tolist())
        self.assertIn('Reference-field Simpson diversity',fig.axes[0].get_xlabel())
        self.assertFalse(any('Standardized mean difference' in a.get_ylabel() for a in fig.axes))
        plt.close(fig)
    def test_five_figure_canvas_and_panel_geometry(self):
        heights=[175,190,170,205,165]
        counts=[7,7,6,8,7]
        for fn,height,n in zip([figure1,figure2,figure3,figure4,figure5],heights,counts):
            seen={}
            with patch('atlas.figures.save',side_effect=lambda fig,c,name,caption:seen.setdefault(name,fig)):
                fn(self.c)
            fig=next(iter(seen.values()))
            np.testing.assert_allclose(fig.get_size_inches(),[180/25.4,height/25.4])
            panels=[a for a in fig.axes if any(t.get_text() in ['a','b','c','c(i)','c(ii)','d','e','f','g'] for t in a.texts)]
            self.assertEqual(len(panels),n)
            self.assertEqual(panels[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().nrows,
                             4 if fn in [figure2,figure4] else 3)
            if fn==figure4:
                self.assertEqual(panels[-1].name,'3d')
                self.assertEqual((panels[-1].elev,panels[-1].azim),(26,-130))
            plt.close(fig)

    def test_public_figure_terminology(self):
        import re
        for fn in [figure2,figure5]:
            seen={}
            with patch('atlas.figures.save',side_effect=lambda fig,c,name,caption:seen.setdefault(name,fig)):fn(self.c)
            fig=next(iter(seen.values()))
            texts=[]
            for a in fig.axes:
                texts += [a.get_xlabel(),a.get_ylabel(),a.get_title()]
                texts += [t.get_text() for t in a.texts+a.get_xticklabels()+a.get_yticklabels()]
                if a.get_legend():texts += [t.get_text() for t in a.get_legend().get_texts()]
            self.assertIsNone(re.search(r'\b(classified|known|nonai)\b',' '.join(texts),re.I))
            plt.close(fig)

if __name__=='__main__':unittest.main()
