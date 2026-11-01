import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from completion_pipeline import build_sessions, join_events, qualifying_interactions, target_sets
from core import ap_at_5


class CompletionContractChecks(unittest.TestCase):
    def test_channel_change_breaks_sitting(self):
        raw = pd.DataFrame({'u':[0]*3,'ch':[3,4,3],'s':[0,40,60],'e':[40,60,100],'raw_id':[0,1,2]})
        sessions, _, _ = build_sessions(raw)
        self.assertEqual(sessions[['ch','s','e']].values.tolist(),[[3,0,40],[4,40,60],[3,60,100]])
        schedule=pd.DataFrame({'ch':[3], 's':[0], 'e':[100], 'item':[12], 'airing':[0]})
        events,_=join_events(sessions,schedule)
        primary,_=qualifying_interactions(events,0,100)
        aggregate,_=qualifying_interactions(events,0,100,aggregate=True)
        self.assertEqual(len(primary),0)
        self.assertEqual(aggregate['count'].tolist(),[1])

    def test_duplicate_overlap_and_switch_clip(self):
        raw = pd.DataFrame({'u':[0]*4,'ch':[3,3,3,4],'s':[0,0,20,80],'e':[70,70,100,110],'raw_id':[0,1,2,3]})
        sessions,_,audit=build_sessions(raw)
        self.assertEqual(sessions[['ch','s','e']].values.tolist(),[[3,0,80],[4,80,110]])
        self.assertEqual(audit['exact_duplicates_removed'],1)
        schedule=pd.DataFrame({'ch':[3], 's':[0], 'e':[100], 'item':[12], 'airing':[0]})
        events,_=join_events(sessions,schedule)
        qual,_=qualifying_interactions(events,0,200)
        self.assertEqual(qual['count'].tolist(),[1])

    def test_full_denominator_at_split_and_two_airings(self):
        sessions=pd.DataFrame({'u':[0], 'ch':[3], 's':[0], 'e':[200], 'session':[0]})
        schedule=pd.DataFrame({'ch':[3,3], 's':[0,100], 'e':[100,200], 'item':[12,12], 'airing':[0,1]})
        events,_=join_events(sessions,schedule)
        before,_=qualifying_interactions(events,0,20)
        after,_=qualifying_interactions(events,20,200)
        self.assertEqual(len(before),0)
        self.assertEqual(after['count'].tolist(),[2])
        self.assertEqual(after['qualified_seconds'].tolist(),[180])

    def test_top_five_ties_and_complete_denominator(self):
        x=pd.DataFrame({'u':[0]*6,'item':[6,5,4,3,2,1],'count':[1]*6})
        rel,_=target_sets(x)
        self.assertEqual(rel[0],{1,2,3,4,5})
        # Three labels, only one retrievable: never shrink denominator to one.
        self.assertAlmostEqual(ap_at_5({1,2,3},[1,8,9,10,11]),1/3)


if __name__ == '__main__':
    unittest.main()
