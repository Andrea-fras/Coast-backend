"""The study streak: which days count and when a streak is alive; offline.

    python3 -m unittest scripts.test_streak
"""
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import _local_day, study_streak  # noqa: E402

TODAY = date(2026, 9, 30)


def days(*offsets):
    return {TODAY - timedelta(days=k) for k in offsets}


class Streak(unittest.TestCase):
    def test_counts_back_from_today(self):
        self.assertEqual(study_streak(days(0, 1, 2), TODAY), 3)

    def test_yesterdays_streak_stays_alive_until_midnight(self):
        self.assertEqual(study_streak(days(1, 2, 3), TODAY), 3)

    def test_a_missed_day_ends_it(self):
        self.assertEqual(study_streak(days(2, 3), TODAY), 0)
        self.assertEqual(study_streak(days(0, 2, 3), TODAY), 1)

    def test_nothing_studied(self):
        self.assertEqual(study_streak(set(), TODAY), 0)


class LocalDay(unittest.TestCase):
    def test_late_evening_in_rome_is_still_today(self):
        # 23:30 in Rome (UTC+2) is 21:30 UTC; 00:30 in Rome is still the previous UTC day.
        self.assertEqual(_local_day(datetime(2026, 9, 30, 21, 30), 120), date(2026, 9, 30))
        self.assertEqual(_local_day(datetime(2026, 9, 29, 22, 30), 120), date(2026, 9, 30))

    def test_aware_and_naive_timestamps_agree(self):
        naive = datetime(2026, 9, 30, 23, 0)
        self.assertEqual(_local_day(naive, -300), _local_day(naive.replace(tzinfo=timezone.utc), -300))
        self.assertEqual(_local_day(naive, -300), date(2026, 9, 30))


if __name__ == "__main__":
    unittest.main()
