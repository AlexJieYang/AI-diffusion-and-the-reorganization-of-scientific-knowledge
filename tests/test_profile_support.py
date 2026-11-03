import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from atlas.knowledge import profile_comparisons
from atlas.describe import PROFILE_MEASURES


class SourceProfileChecks(unittest.TestCase):
    def test_pooled_paper_weights_original_units_and_changing_support(self):
        rows = []
        for year in [2023,2024]:
            for sid in [1,2]:
                for group in [0,1]:
                    n = 1 if (year,sid,group)==(2024,2,1) else 10
                    mean = {1:[.2,.5],2:[.7,.9]}[sid][group]
                    row = dict(year=year,source_id=sid,ai_group=group,n=n)
                    for metric in PROFILE_MEASURES:
                        row.update({metric+'_n':n,metric+'_mean':mean,metric+'_coverage':1.})
                    rows.append(row)
        source = pd.DataFrame(rows)
        catalog = pd.DataFrame({'source_id':[1,2],'source_label':['S1','S2']})
        with tempfile.TemporaryDirectory() as temp:
            c = dict(output_dir=str(Path(temp)/'results'),trend_start=2023,end_year=2024,min_profile_cell_n=5)
            def read(c,name):
                return catalog.copy() if name=='source_catalog.csv' else source.copy()
            with patch('atlas.knowledge.read_table',side_effect=read):
                result = profile_comparisons(c)
        z = result[result.metric.eq('ref_field_simpson')].set_index('year')
        np.testing.assert_allclose(z.source_adjusted_difference,[.25,.3])
        self.assertEqual(z.n_sources.tolist(),[2,1])
        self.assertEqual(z.support_changed.tolist(),[False,True])
        self.assertEqual(z.support_segment.tolist(),[0,1])
        self.assertAlmostEqual(z.loc[2024,'ai_positive_retained_fraction'],10/11)
        self.assertAlmostEqual(z.loc[2024,'other_scored_retained_fraction'],.5)
        np.testing.assert_allclose(z.raw_difference,z.within_component+z.composition_component)


if __name__=='__main__':
    unittest.main()
