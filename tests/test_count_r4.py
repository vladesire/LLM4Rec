import sys,itertools,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from count_r4 import moments,expected_ap,greedy,sampled_targets
from core import ap_at_5
class CountDecisionChecks(unittest.TestCase):
 def test_identity_short_empty_and_exhaustive(self):
  t=np.array([[1,2,-1,-1,-1],[-1]*5,[2,3,4,-1,-1],[1,3,5,6,-1]])
  ids=np.arange(1,7);p,a,b=moments(t,ids)
  for order in itertools.permutations(range(6),5):
   direct=np.mean([ap_at_5(set(r[r>=0]),ids[list(order)]) for r in t])
   self.assertAlmostEqual(expected_ap(order,a,b),direct)
  g=greedy(a,b,ids);opt=max(expected_ap(o,a,b) for o in itertools.permutations(range(6),5))
  self.assertLessEqual(expected_ap(g,a,b),opt+1e-12)
 def test_counts_bounded_and_zero_targets(self):
  t,n=sampled_targets(np.ones(6),np.ones(6),np.zeros(6,dtype=int),np.arange(1,7),512,42,1)
  self.assertTrue((n==0).all());self.assertTrue((t==-1).all())
if __name__=='__main__':unittest.main()
