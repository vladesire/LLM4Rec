import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from data_pipeline import canonical_schedule, labels, match_window


class JoinIntegrationChecks(unittest.TestCase):
    def test_clipping_examples_and_threshold(self):
        views = pd.DataFrame({'u': [0, 1], 'ch': [3, 3], 's': [50, 140], 'e': [250, 220]})
        schedule = pd.DataFrame({'ch': [3, 3], 's': [100, 200], 'e': [200, 300], 'item': [12, 13], 'airing': [0, 1]})
        events, audit = match_window(views, schedule, 0, 200, 'test fitting')
        self.assertEqual(events.seconds.tolist(), [100, 60])
        self.assertEqual(audit['labelled_matched_seconds'], 160)
        self.assertEqual(audit['examples'][0]['item'], 12)
        interactions, counts = labels(events, 75)
        self.assertEqual(interactions.item.tolist(), [12])
        self.assertEqual(counts['user_airings_dropped_below_threshold'], 1)

    def test_conflicts_excluded_and_duplicates_removed(self):
        schedule = pd.DataFrame({'ch': [3]*4, 's': [0, 0, 90, 200], 'e': [100, 100, 150, 300], 'item': [12, 12, 13, 14]})
        clean, audit = canonical_schedule(schedule)
        self.assertEqual(clean.item.tolist(), [14])
        self.assertEqual(audit['duplicate_rows_removed'], 1)
        self.assertEqual(audit['overlapping_airings_removed'], 2)


if __name__ == '__main__':
    unittest.main()
