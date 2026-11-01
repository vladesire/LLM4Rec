import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from behavioral_r2 import decay_weights,effective_days,habit_cell,features


class BehavioralFeatureChecks(unittest.TestCase):
    def test_recency_and_future_guard(self):
        now=100*86400
        np.testing.assert_array_equal(decay_weights([now,now-28*86400],now,28),[1,.5])
        np.testing.assert_array_equal(decay_weights([now,now-28*86400],now,None),[1,1])
        with self.assertRaises(AssertionError):decay_weights([now+1],now,28)

    def test_normalized_history_length(self):
        self.assertEqual(effective_days(84,None),84)
        self.assertAlmostEqual(effective_days(28,28),28/np.log(2)*.5)
        self.assertLess(effective_days(84,28),effective_days(84,56))

    def test_weekday_and_time_cells(self):
        # Monday midnight and Saturday noon on the same channel.
        seconds=np.array(['2020-06-01T00:00:00','2020-06-06T12:00:00'],dtype='datetime64[s]').astype('int64')
        np.testing.assert_array_equal(habit_cell([3,3],seconds),[24,30])

    def test_forecasts_shrinkage_caps_and_cold_fallback(self):
        cutoff=20*86400
        watches=pd.DataFrame({'u':[0,0],'item':[10,10],'ch':[3,3],'cell':[24,24],
                              'overlap_end':[cutoff-100,cutoff-200]})
        history=pd.DataFrame({'item':[10,10,10,10,11],'ch':[3]*5,'cell':[24]*5,
                              's':[cutoff-1000]*5,'e':[cutoff-1]*5})
        target=pd.DataFrame({'item':[10,11,11,11,11],'ch':[3]*5,'cell':[24]*5})
        ctx={'watch':watches,'historical':history,'target':target,'users':[0,1],
             'candidates':np.array([10,11]),'cutoff':cutoff,'history_days':10,'forecast_days':2,
             'config':{'opportunity_prior_airings':2,'opportunity_ratio_bounds':[.25,4],
                       'habit_prior_airings':20}}
        result=features(ctx,None)
        np.testing.assert_allclose(result['repeat'],[[.4,0],[0,0]])
        np.testing.assert_allclose(result['opportunity'],[[.5,0],[0,0]])
        np.testing.assert_allclose(result['capped_opportunity'],[[1,0],[0,0]])
        np.testing.assert_allclose(result['channel_habit'],[[.4,1.6],[0,0]])
        np.testing.assert_allclose(result['time_habit'],[[.4,1.6],[0,0]])
        self.assertEqual(result['details']['opportunity_ratio_max'],4)


if __name__=='__main__':unittest.main()
