import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from atlas.figures import fixed_support_panel,pair_distance_panel


class Figure3Support(unittest.TestCase):
    def setUp(self):
        self.fig,self.ax=plt.subplots()
        self.config=dict(fixed_support_start=2000,end_year=2024,
                         min_country_period_mass=200,min_ai_annual_mass=10)

    def tearDown(self):
        plt.close(self.fig)

    def test_empty_fixed_support_is_explicit(self):
        self.assertEqual(fixed_support_panel(self.ax,pd.DataFrame(),self.config),0)
        self.assertEqual(len(self.ax.lines),0)
        self.assertIn('0 eligible countries',self.ax.texts[0].get_text())
        self.assertIn('2000-2024',self.ax.texts[0].get_text())

    def test_one_country_is_not_a_between_country_panel(self):
        fixed=pd.DataFrame([dict(year=2024,metric='ai',effective_between=1,
                                 n_countries=1)])
        self.assertEqual(fixed_support_panel(self.ax,fixed,self.config),1)
        self.assertEqual(len(self.ax.lines),0)

    def test_supported_fixed_panel_retains_values(self):
        fixed=pd.DataFrame([dict(year=y,metric=m,effective_between=v,
                                 n_countries=3)
                            for m in ['ai','nonai','known']
                            for y,v in [(2020,1.1),(2024,1.3)]])
        before=fixed.copy(deep=True)
        self.assertEqual(fixed_support_panel(self.ax,fixed,self.config),3)
        self.assertEqual(len(self.ax.lines),3)
        for line in self.ax.lines:np.testing.assert_array_equal(line.get_ydata(),[1.1,1.3])
        pd.testing.assert_frame_equal(fixed,before)

    def test_empty_pairs_do_not_draw_a_zero_density(self):
        support=pair_distance_panel(self.ax,pd.DataFrame())
        self.assertTrue(all(r['n_pairs']==0 and r['display']=='unavailable' for r in support))
        self.assertEqual(len(self.ax.collections),0)
        self.assertEqual(len(self.ax.lines),0)
        self.assertEqual(sum('No eligible pairs' in t.get_text() for t in self.ax.texts),4)

    def test_sparse_constant_and_dense_pairs_are_distinguished(self):
        values={2000:[.1,.2,.3],2010:[0]*6,2020:[.1,.2,.3,.4,.5]}
        rows=[dict(year=y,metric='ai',jsd=v) for y,vs in values.items() for v in vs]
        rows.extend([dict(year=2024,metric='ai',jsd=np.nan),
                     dict(year=2024,metric='known',jsd=.5)])
        js=pd.DataFrame(rows);before=js.copy(deep=True)
        support=pair_distance_panel(self.ax,js)
        self.assertEqual([r['n_pairs'] for r in support],[3,6,5,0])
        self.assertEqual([r['display'] for r in support],
                         ['observations','observations','density','unavailable'])
        np.testing.assert_array_equal(self.ax.lines[0].get_xdata(),values[2000])
        np.testing.assert_array_equal(self.ax.lines[1].get_xdata(),values[2010])
        self.assertEqual(len(self.ax.collections),1)
        pd.testing.assert_frame_equal(js,before)


if __name__=='__main__':unittest.main()
