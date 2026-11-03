import unittest
import numpy as np
import pandas as pd
from atlas.windowed import windowed_portfolios,common_source_support


class WindowedPortfolios(unittest.TestCase):
    def setUp(self):
        self.c=dict(trend_start=2000,end_year=2024,fixed_support_start=2000,
                    baseline_years=[2010,2014],portfolio_window_years=5,
                    standardization_window_years=5,min_country_period_mass=20,
                    min_cell_period_mass=5,min_ai_period_mass=3)
        rows=[]
        for year in range(2000,2025):
            for country in ['AA','BB']:
                for source_id in [0,1]:
                    active=(year%5==0 and source_id==(0 if country=='AA' else 1)) or (
                            year%5==1 and source_id==(1 if country=='AA' else 0))
                    ai=3. if active else 0.
                    rows.append(dict(year=year,country=country,source_id=source_id,known=3.,
                                     ai=ai,nonai=3.-ai,score=ai,ai_gt02=ai,ai_gt08=ai))
        self.annual=pd.DataFrame(rows)

    def test_pooling_recovers_a_fixed_panel_from_sparse_years(self):
        before=self.annual.copy(deep=True)
        frames=windowed_portfolios(self.annual,self.c)
        countries=frames['diversity_fixed_countries_period']
        self.assertEqual(set(countries.country),{'AA','BB'})
        np.testing.assert_allclose(countries.baseline_known_weight,[.5,.5])
        fixed=frames['diversity_fixed_support_period']
        self.assertEqual(len(fixed),15)
        self.assertTrue(fixed.n_countries.eq(2).all())
        support=frames['diversity_period_support']
        self.assertTrue(support.n_pairs.eq(1).all())
        np.testing.assert_allclose(support.ai_fractional_credit,12)
        pd.testing.assert_frame_equal(before,self.annual)

    def test_entropy_is_calculated_from_pooled_credit(self):
        q=windowed_portfolios(self.annual,self.c)['diversity_period_decomposition']
        q=q[q.metric.eq('ai')&q.weighting.eq('mass')]
        # Each active annual portfolio is single-source; the pooled portfolio has both sources.
        np.testing.assert_allclose(q.within_h,np.log(2))
        np.testing.assert_allclose(q.global_h,q.within_h+q.between_h,atol=1e-12)
        self.assertTrue((q.global_h<=np.log(2)+1e-12).all())

    def test_common_source_support_uses_every_window(self):
        self.annual['known']=1.2
        support=common_source_support(self.annual,self.c)
        self.assertEqual(support,{('AA',0),('AA',1),('BB',0),('BB',1)})
        self.annual.loc[self.annual.country.eq('AA')&self.annual.source_id.eq(0)&self.annual.year.between(2005,2009),'known']=0
        self.assertNotIn(('AA',0),common_source_support(self.annual,self.c))

    def test_missing_window_does_not_become_supported(self):
        partial=self.annual[~(self.annual.country.eq('AA')&self.annual.year.between(2005,2009))]
        frames=windowed_portfolios(partial,self.c)
        self.assertEqual(set(frames['diversity_fixed_countries_period'].country),{'BB'})

    def test_target_year_zero_does_not_drop_a_countrys_other_sources(self):
        self.annual.loc[self.annual.country.eq('AA')&self.annual.source_id.eq(0)&self.annual.year.eq(2024),'known']=0
        support=common_source_support(self.annual,self.c)
        self.assertNotIn(('AA',0),support)
        self.assertIn(('AA',1),support)


if __name__=='__main__':unittest.main()
