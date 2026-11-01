import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from core import ap_at_5, clip_interval, match_channel, overlap_seconds, top_five


class FocusedChecks(unittest.TestCase):
    def test_interval_cases(self):
        self.assertEqual(overlap_seconds(90, 150, 100, 200), 50)
        self.assertEqual(overlap_seconds(0, 100, 100, 200), 0)
        vi, ai, sec = match_channel([90, 200, 50], [150, 300, 250], [100, 200], [200, 300])
        self.assertEqual(sorted(zip(vi, ai, sec)), [(0, 0, 50), (1, 1, 100), (2, 0, 100), (2, 1, 50)])

    def test_split_crossing(self):
        a, b = clip_interval(90, 250, 0, 200)
        self.assertEqual(overlap_seconds(a, b, 100, 300), 100)
        a, b = clip_interval(90, 250, 200, 300)
        self.assertEqual(overlap_seconds(a, b, 100, 300), 50)

    def test_ap(self):
        self.assertAlmostEqual(ap_at_5({'A', 'B', 'C'}, ['A', 'B', 'D', 'C', 'E']), 11 / 12)
        self.assertAlmostEqual(ap_at_5({'A', 'B', 'C'}, ['D', 'A', 'E', 'B', 'C']), 8 / 15)
        self.assertEqual(ap_at_5(set(), ['A', 'B', 'C', 'D', 'E']), 0)
        with self.assertRaises(ValueError):
            ap_at_5({'A'}, ['A', 'A', 'B', 'C', 'D'])

    def test_common_fallback_and_ties(self):
        ids, fallback = top_five([1, 2, 3, 4, 5, 6], [0, 0, 2, 2, 0, 0], [10, 9, 8, 7, 6, 5])
        self.assertEqual(ids, [3, 4, 1, 2, 5])
        self.assertEqual(fallback, [False, False, True, True, True])


if __name__ == '__main__':
    unittest.main()
