"""Regression tests for the fixed-alarm minimum-duration audit."""
import csv
from pathlib import Path
import sys
import unittest

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
from wn_duration_sensitivity import (
    DURATION_CSV, HISTOGRAM_CSV,
    assess, consistent_records, episode_lengths,
)


class DurationSensitivityTests(unittest.TestCase):
    def test_initial_fragmentation_not_an_onset_and_trailing_length_is_a_lower_bound(self):
        truth = dict(enumerate((2, 2, 1, 2, 1, 1, 3, 2)))
        self.assertEqual(episode_lengths(truth), [(3, 1, False), (6, 2, True)])

    def test_fixed_alarm_can_be_rematched_to_a_later_qualifying_onset(self):
        episodes = [(5, 1, False), (8, 3, False)]
        self.assertEqual(assess(episodes, [4], 1)[1], [1])
        revised, leads = assess(episodes, [4], 3)
        self.assertEqual(leads, [4])
        self.assertEqual(revised["confirmed_shorter_events"], 1)
        self.assertEqual(revised["unmatched_resolved_alarms"], 0)

    def test_right_censoring_does_not_imply_short_or_false(self):
        episodes = [(8, 1, True)]
        c, leads = assess(episodes, [2], 2)
        self.assertEqual(leads, [])
        self.assertEqual(c["right_censored_unconfirmed_events"], 1)
        self.assertEqual(c["unresolved_alarms"], 1)
        self.assertEqual(c["unmatched_resolved_alarms"], 0)
        self.assertEqual(assess(episodes, [1], 2)[0]["unmatched_resolved_alarms"], 1)
        self.assertEqual(assess([(8, 2, True)], [2], 2)[1], [6])

    def test_one_to_one_and_resolved_false_alarm_accounting(self):
        c, leads = assess([(8, 2, True)], [2, 3], 2)
        self.assertEqual(leads, [6])
        self.assertEqual(c["matched_events"], 1)
        self.assertEqual(c["unmatched_resolved_alarms"], 1)

    def test_verified_records_and_released_csv(self):
        summary, histogram = consistent_records()
        self.assertEqual(
            [(r["confirmed_events"], r["right_censored_unconfirmed_events"], r["matched_events"],
              r["unmatched_resolved_alarms"], r["unresolved_alarms"]) for r in summary],
            [(1114, 0, 183, 88, 0), (515, 7, 136, 135, 0), (317, 8, 105, 166, 0)],
        )
        self.assertEqual([r["emitted_alarms"] for r in summary], [271] * 3)
        self.assertEqual([r["median_lead_seconds"] for r in summary], [.2] * 3)
        self.assertEqual(histogram[0], {"observed_disconnected_frames": 1,
                                        "ended_before_record_end": 592, "right_censored": 7})
        self.assertEqual(sum(r["right_censored"] for r in histogram), 18)
        for path, values in [(DURATION_CSV, summary), (HISTOGRAM_CSV, histogram)]:
            with path.open(newline="", encoding="utf-8") as f:
                self.assertEqual(list(csv.DictReader(f)),
                                 [{k: str(v) if v is not None else "" for k, v in r.items()} for r in values])
