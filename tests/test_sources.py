import unittest
import numpy as np
import pandas as pd
from atlas.sources import source_ids, build_source_catalog
from atlas.runtime import DEFAULT


class SourceChecks(unittest.TestCase):
    def test_exact_source_ids(self):
        values = [str(2**53+1), 'https://openalex.org/S9007199254740995']
        self.assertEqual(source_ids(pd.Series(values)).tolist(), [2**53+1, 2**53+3])
        with self.assertRaises(ValueError):
            source_ids(pd.Series([float(2**53+2)]))

    def test_examples_rank_by_volume_and_do_not_select_the_population(self):
        c = dict(DEFAULT, min_source_group_n=2, representative_source_n=2)
        m = pd.DataFrame({'source_id':[1]*10+[2]*8+[3]*7,
            'AIProb':[.9]*2+[.1]*8+[.9]*6+[.1]*2+[.1]*7, 'year':[2024]*25})
        before = m.copy(deep=True)
        s = build_source_catalog(m,c)
        self.assertEqual(s.loc[s.selected,'source_id'].tolist(),[1,2])
        self.assertEqual(s.all_scored.sum(),len(m))
        self.assertFalse(s.set_index('source_id').loc[3,'example_eligible'])
        pd.testing.assert_frame_equal(m,before)

    def test_explicit_sources_need_the_same_support(self):
        c = dict(DEFAULT, min_source_group_n=2, representative_source_ids=[3])
        m = pd.DataFrame({'source_id':[3]*5, 'AIProb':[.1]*5, 'year':[2024]*5})
        with self.assertRaisesRegex(ValueError,'support'):
            build_source_catalog(m,c)

    def test_no_eligible_examples_are_not_fabricated(self):
        m = pd.DataFrame({'source_id':[1,1], 'AIProb':[0.,1.], 'year':[2024,2024]})
        s = build_source_catalog(m,DEFAULT)
        self.assertEqual(s.selected.sum(),0)


if __name__ == '__main__':
    unittest.main()
