import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from atlas.runtime import exact_ids,hash64,SingleWriter
from atlas.portfolios import rca_matrix,diversity_identity,kitagawa,pairwise_distances,add_rcas

class MetricChecks(unittest.TestCase):
    def test_exact_ids(self):
        raw=pd.Series([str(2**53+1),'W9007199254740995'])
        ids=exact_ids(raw)
        self.assertEqual(int(ids[0]),2**53+1);self.assertEqual(int(ids[1]),2**53+3)
        with self.assertRaises(ValueError):exact_ids(pd.Series([float(2**53+2)]))
        with self.assertRaises(ValueError):exact_ids(pd.Series([-1]))
    def test_rca(self):
        x=np.array([[20.,80.],[0.,100.]])
        r=rca_matrix(x);self.assertAlmostEqual(r[0,0],2);self.assertEqual(r[1,0],0)
        self.assertTrue(np.isnan(rca_matrix(np.zeros((2,2)))).all())
    def test_entropy_identity(self):
        x=np.array([[10.,0.],[0.,10.]])
        r=diversity_identity(x);self.assertAlmostEqual(r['within_h'],0)
        self.assertAlmostEqual(r['between_h'],np.log(2));self.assertAlmostEqual(r['effective_between'],2)
        self.assertAlmostEqual(diversity_identity(np.ones((4,3)))['between_h'],0)
    def test_accounting_entry_exit(self):
        a=pd.DataFrame({'denom':[20.,10.],'numer':[4.,7.]},index=['common','exit'])
        b=pd.DataFrame({'denom':[10.,30.],'numer':[5.,6.]},index=['common','entry'])
        z=kitagawa(a,b).set_index('component').contribution
        self.assertAlmostEqual(z.drop('total').sum(),z['total'])
    def test_hash_chunk_invariance(self):
        a=np.arange(1,100,dtype='uint64')
        np.testing.assert_array_equal(hash64(a),np.r_[hash64(a[:30]),hash64(a[30:])])
    def test_sparse_source_distances_match_probability_definition(self):
        from scipy.sparse import csr_matrix
        from scipy.spatial.distance import jensenshannon
        x=np.array([[4.,0.,2.,0.],[0.,3.,1.,0.],[1.,0.,0.,5.]])
        rows=list(pairwise_distances(csr_matrix(x),['AA','BB','CC'],1))
        p=x/x.sum(1,keepdims=True)
        expected=[jensenshannon(p[i],p[j])**2 for i,j in [(0,1),(0,2),(1,2)]]
        np.testing.assert_allclose([r['jsd'] for r in rows],expected,atol=1e-12)
    def test_source_rca_preserves_observed_zero_and_sparse_cells(self):
        rows=[]
        for country,sid,known,ai in [('AA',1,20.,4.),('AA',2,80.,0.),('BB',2,100.,10.)]:
            rows.append(dict(year=2024,country=country,source_id=sid,known=known,ai=ai,
                nonai=known-ai,score=ai,ai_gt02=ai,ai_gt08=ai))
        result=add_rcas(pd.DataFrame(rows))
        self.assertEqual(len(result),3)
        zero=result[(result.country=='AA')&(result.source_id==2)].iloc[0]
        self.assertEqual(zero.rca_ai,0)
        self.assertEqual(zero.rsca_ai,-1)
        self.assertFalse(((result.country=='BB')&(result.source_id==1)).any())
    def test_single_file_writer(self):
        with tempfile.TemporaryDirectory() as q:
            target=Path(q)/'one.parquet'
            with SingleWriter(target) as w:
                w.write(pd.DataFrame({'x':[1,2]}));w.write(pd.DataFrame({'x':[3]}))
            w.close()
            self.assertEqual(len(list(Path(q).glob('*.parquet'))),1)
            self.assertEqual(pd.read_parquet(target).x.tolist(),[1,2,3])

if __name__=='__main__':unittest.main()
