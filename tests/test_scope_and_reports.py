import unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from atlas.runtime import fresh_outputs
from atlas.publication_tables import report_frames
from atlas.describe import group_codes

class ScopeAndReports(unittest.TestCase):
    def test_missing_scores_cannot_be_grouped(self):
        with self.assertRaises(ValueError):group_codes([.1,np.nan,.9])
        np.testing.assert_array_equal(group_codes([0,.5,1]),[0,0,1])
    def test_reports_select_one_year(self):
        rows=[dict(year=y,metric=k,n_countries=10,gini=.4,hhi=.2,top5_share=.99 if y==2020 else .65,top10_share=1) for y in [2020,2024] for k in ['known','ai','nonai','score']]
        standard=pd.DataFrame([dict(year=y,country='AA',n_sources=2,retained_baseline_source_mass=.5,raw_common_share=.9 if y==2020 else .01,standardized_share=.8 if y==2020 else .02) for y in [2020,2024]])
        t7,t13,bars,points=report_frames(pd.DataFrame(rows),standard)
        np.testing.assert_allclose(bars.top_five_pct,65)
        self.assertEqual(len(points),1);self.assertEqual(points.iloc[0].raw_ai_positive_pct,1)
        self.assertEqual(points.iloc[0].standardized_ai_positive_pct,2)
    def test_fresh_output_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'input').mkdir();c=dict(output_dir=str(root/'analysis'),research_dir=str(root/'input'))
            d=fresh_outputs(c);(d['tables']/'old.csv').write_text('stale');fresh_outputs(c)
            self.assertFalse((d['tables']/'old.csv').exists())
            with self.assertRaises(ValueError):fresh_outputs(dict(output_dir=str(root/'input'),research_dir=str(root/'input')))
            foreign=root/'unrelated';foreign.mkdir();(foreign/'keep.txt').write_text('keep')
            with self.assertRaises(ValueError):fresh_outputs(dict(output_dir=str(foreign),research_dir=str(root/'input')))
            self.assertTrue((foreign/'keep.txt').is_file())

if __name__=='__main__':unittest.main()
