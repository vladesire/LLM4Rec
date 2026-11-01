import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rerank_features import candidate_pool,candidate_diagnostics,make_snapshot
from completion_pipeline import epoch


class RerankerBoundaryChecks(unittest.TestCase):
    def test_pool_is_distinct_eligible_and_target_independent(self):
        items=np.arange(1,21);base=np.arange(20,0,-1,dtype=float)[None,:]
        scores={'personal_frequency':base,'R2':base[:,::-1],'als':base,'content':base}
        pool=candidate_pool(scores,items,np.zeros_like(base),base[0],
                            {'personal_frequency':2,'R2':5,'als':1,'content':1,'popularity':1},10)
        ids=items[pool[0]]
        self.assertEqual(len(set(ids)),10)
        self.assertTrue(set(ids).issubset(set(items)))
        self.assertTrue({16,17,18,19,20}.issubset(set(ids)))

    def test_oracle_keeps_unretrievable_labels_and_empty_users(self):
        report=candidate_diagnostics(np.array([[1,8,9],[1,2,3]]),{0:{1,2,999}},[0,1])
        self.assertAlmostEqual(report['micro_recall'],1/3)
        self.assertAlmostEqual(report['oracle_map_at_5_ceiling'],1/6)

    def test_late_audit_guard_runs_before_loading(self):
        with self.assertRaises(AssertionError):
            make_snapshot(None,None,epoch('2020-06-29'),epoch('2020-07-27'),None,None,None,label=True)


if __name__=='__main__':unittest.main()
