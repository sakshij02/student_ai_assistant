"""
test_tools.py — Unit tests for tools.py.

Coverage:
  get_weak_topics          — happy path, invalid student, subject inference,
                             score attachment, unknown subject → None
  get_upcoming_tests       — happy path, invalid student, past test filtered,
                             days_remaining value, sort order, empty result,
                             topics list present
  recommend_study_material — semantic hit used, fallback on empty semantic,
                             multiple topics, unknown topic, empty input
  _fallback_match          — exact match, case-insensitive partial, no match,
                             capped at 2 results
  get_study_plan           — invalid student, all weak topics included,
                             test topics included, weak+test = highest score,
                             weak-only scores 5, sorted descending,
                             past test excluded from urgency map,
                             urgency decay for sooner test,
                             daily_study_time and student_name returned
  dispatch_tool            — known tool returns JSON, unknown tool → error JSON,
                             output always valid JSON, all four tools dispatchable

Run with:
    python -m unittest test_tools -v
"""

import json
import sys
import unittest
from datetime import date, timedelta

from test_config import (
    FUTURE_DATE,
    MATERIALS,
    PAST_DATE,
    PERFORMANCE,
    PROFILE,
    TESTS,
    TESTS_WITH_PAST,
    load_tools_with_fixtures,
)


# =============================================================================
# get_weak_topics
# =============================================================================

class TestGetWeakTopics(unittest.TestCase):

    def setUp(self):
        self.t = load_tools_with_fixtures()

    def test_valid_student_returns_all_weak_topics(self):
        result = self.t.get_weak_topics("S123")
        topics = [e["topic"] for e in result["weak_topics"]]
        self.assertEqual(
            topics,
            ["Algebra", "Quadratic Equations", "Light - Reflection and Refraction"],
        )

    def test_invalid_student_returns_error(self):
        result = self.t.get_weak_topics("UNKNOWN")
        self.assertIn("error", result)

    def test_math_topic_inferred_as_mathematics(self):
        result = self.t.get_weak_topics("S123")
        algebra = next(e for e in result["weak_topics"] if e["topic"] == "Algebra")
        self.assertEqual(algebra["subject"], "Mathematics")

    def test_non_math_topic_inferred_as_science(self):
        result = self.t.get_weak_topics("S123")
        light = next(e for e in result["weak_topics"] if "Light" in e["topic"])
        self.assertEqual(light["subject"], "Science")

    def test_math_subject_score_attached_correctly(self):
        result = self.t.get_weak_topics("S123")
        algebra = next(e for e in result["weak_topics"] if e["topic"] == "Algebra")
        self.assertEqual(algebra["subject_score_percentage"], 52)

    def test_science_subject_score_attached_correctly(self):
        result = self.t.get_weak_topics("S123")
        light = next(e for e in result["weak_topics"] if "Light" in e["topic"])
        self.assertEqual(light["subject_score_percentage"], 63)

    def test_strong_topics_present_in_result(self):
        result = self.t.get_weak_topics("S123")
        self.assertEqual(result["strong_topics"], PROFILE["strong_topics"])

    def test_daily_study_time_present_in_result(self):
        result = self.t.get_weak_topics("S123")
        self.assertEqual(result["daily_study_time_minutes"], 90)

    def test_unknown_subject_score_is_none(self):
        """Topic with no math keyword → Science; if no Science score exists → None."""
        mod = load_tools_with_fixtures()
        mod._profile     = {**PROFILE, "weak_topics": ["Topology"]}
        mod._performance = {"student_id": "S123", "subject_performance": []}
        result = mod.get_weak_topics("S123")
        self.assertIsNone(result["weak_topics"][0]["subject_score_percentage"])


# =============================================================================
# get_upcoming_tests
# =============================================================================

class TestGetUpcomingTests(unittest.TestCase):

    def setUp(self):
        self.t = load_tools_with_fixtures()

    def test_valid_student_returns_upcoming_test(self):
        result = self.t.get_upcoming_tests("S123")
        self.assertEqual(result["total_count"], 1)
        self.assertEqual(result["upcoming_tests"][0]["test_id"], "T201")

    def test_invalid_student_returns_error(self):
        result = self.t.get_upcoming_tests("UNKNOWN")
        self.assertIn("error", result)

    def test_past_test_is_filtered_out(self):
        mod = load_tools_with_fixtures(tests_data=TESTS_WITH_PAST)
        result = mod.get_upcoming_tests("S123")
        ids = [t["test_id"] for t in result["upcoming_tests"]]
        self.assertNotIn("T200", ids)
        self.assertIn("T201", ids)

    def test_days_remaining_value_is_correct(self):
        result = self.t.get_upcoming_tests("S123")
        self.assertEqual(result["upcoming_tests"][0]["days_remaining"], 6)

    def test_tests_sorted_soonest_first(self):
        further_date = (date.today() + timedelta(days=20)).strftime("%Y-%m-%d")
        two_tests = {
            "student_id": "S123",
            "upcoming_tests": [
                {"test_id": "T202", "subject": "Science",
                 "test_name": "Science Test", "date": further_date, "topics": ["Light"]},
                {"test_id": "T201", "subject": "Mathematics",
                 "test_name": "Math Test", "date": FUTURE_DATE,   "topics": ["Algebra"]},
            ],
        }
        mod = load_tools_with_fixtures(tests_data=two_tests)
        result = mod.get_upcoming_tests("S123")
        self.assertEqual(result["upcoming_tests"][0]["test_id"], "T201")
        self.assertEqual(result["upcoming_tests"][1]["test_id"], "T202")

    def test_no_upcoming_tests_returns_empty_list(self):
        all_past = {
            "student_id": "S123",
            "upcoming_tests": [
                {"test_id": "T200", "subject": "Math", "test_name": "Old",
                 "date": PAST_DATE, "topics": ["Algebra"]},
            ],
        }
        mod = load_tools_with_fixtures(tests_data=all_past)
        result = mod.get_upcoming_tests("S123")
        self.assertEqual(result["total_count"], 0)
        self.assertEqual(result["upcoming_tests"], [])

    def test_topics_list_included_in_each_entry(self):
        result = self.t.get_upcoming_tests("S123")
        self.assertIn("Algebra", result["upcoming_tests"][0]["topics"])


# =============================================================================
# recommend_study_material
# =============================================================================

class TestRecommendStudyMaterial(unittest.TestCase):

    def setUp(self):
        self.t = load_tools_with_fixtures()

    def test_semantic_results_used_when_available(self):
        semantic_hit = [{"material_id": "M101", "topic": "Algebra",
                         "title": "Algebra Basics Revision Notes"}]
        sys.modules["embeddings"].search_materials = lambda topic, top_k=2: semantic_hit
        result = self.t.recommend_study_material(["Algebra"])
        self.assertEqual(result["recommendations"]["Algebra"], semantic_hit)

    def test_fallback_used_when_semantic_returns_empty(self):
        # embeddings stub already returns [] by default from load_tools_with_fixtures
        result = self.t.recommend_study_material(["Algebra"])
        recs = result["recommendations"]["Algebra"]
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0]["material_id"], "M101")

    def test_multiple_topics_each_get_recommendations(self):
        result = self.t.recommend_study_material(["Algebra", "Quadratic Equations"])
        self.assertIn("Algebra", result["recommendations"])
        self.assertIn("Quadratic Equations", result["recommendations"])

    def test_unknown_topic_returns_empty_list(self):
        result = self.t.recommend_study_material(["Thermodynamics"])
        self.assertEqual(result["recommendations"]["Thermodynamics"], [])

    def test_empty_topics_list_returns_empty_recommendations(self):
        result = self.t.recommend_study_material([])
        self.assertEqual(result["recommendations"], {})


# =============================================================================
# _fallback_match
# =============================================================================

class TestFallbackMatch(unittest.TestCase):

    def setUp(self):
        self.t = load_tools_with_fixtures()

    def test_exact_topic_match(self):
        result = self.t._fallback_match("Algebra")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["material_id"], "M101")

    def test_case_insensitive_partial_match(self):
        result = self.t._fallback_match("algebra")
        self.assertGreater(len(result), 0)

    def test_no_match_returns_empty_list(self):
        result = self.t._fallback_match("Thermodynamics")
        self.assertEqual(result, [])

    def test_capped_at_two_results(self):
        self.t._materials_raw = {
            "materials": [
                {"material_id": "M1", "topic": "Algebra", "title": "Notes 1"},
                {"material_id": "M2", "topic": "Algebra", "title": "Notes 2"},
                {"material_id": "M3", "topic": "Algebra", "title": "Notes 3"},
            ]
        }
        result = self.t._fallback_match("Algebra")
        self.assertLessEqual(len(result), 2)


# =============================================================================
# get_study_plan
# =============================================================================

class TestGetStudyPlan(unittest.TestCase):

    def setUp(self):
        self.t = load_tools_with_fixtures()

    def test_invalid_student_returns_error(self):
        result = self.t.get_study_plan("UNKNOWN")
        self.assertIn("error", result)

    def test_result_contains_study_plan_key(self):
        result = self.t.get_study_plan("S123")
        self.assertIn("study_plan", result)

    def test_all_weak_topics_appear_in_plan(self):
        result = self.t.get_study_plan("S123")
        topics_in_plan = {e["topic"] for e in result["study_plan"]}
        for wt in PROFILE["weak_topics"]:
            self.assertIn(wt, topics_in_plan)

    def test_test_topics_appear_in_plan(self):
        result = self.t.get_study_plan("S123")
        topics_in_plan = {e["topic"] for e in result["study_plan"]}
        self.assertIn("Algebra", topics_in_plan)
        self.assertIn("Quadratic Equations", topics_in_plan)

    def test_weak_and_test_topic_gets_highest_score(self):
        """Topics that are both weak (+5) and in a test (+10 + urgency) outrank weak-only topics."""
        result = self.t.get_study_plan("S123")
        first = result["study_plan"][0]
        # Both Algebra and Quadratic Equations share the same score (both weak + same test),
        # so we assert on the invariant rather than which one wins the tie.
        self.assertTrue(first["is_weak"])
        self.assertTrue(first["in_upcoming_test"])
        self.assertGreater(first["priority_score"], 5.0)

    def test_weak_only_topic_scores_exactly_5(self):
        """Light is weak (+5) but not in any test → score must be 5.0."""
        result = self.t.get_study_plan("S123")
        light = next(e for e in result["study_plan"] if "Light" in e["topic"])
        self.assertEqual(light["priority_score"], 5.0)
        self.assertFalse(light["in_upcoming_test"])

    def test_plan_sorted_descending_by_priority_score(self):
        result = self.t.get_study_plan("S123")
        scores = [e["priority_score"] for e in result["study_plan"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_past_test_excluded_from_urgency_map(self):
        mod = load_tools_with_fixtures(tests_data=TESTS_WITH_PAST)
        result = mod.get_study_plan("S123")
        algebra = next(e for e in result["study_plan"] if e["topic"] == "Algebra")
        for test in algebra["upcoming_tests"]:
            self.assertGreaterEqual(test["days_remaining"], 0)

    def test_urgency_bonus_higher_for_sooner_test(self):
        """A topic in a 2-day test must outscore the same topic in a 15-day test."""
        soon  = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
        later = (date.today() + timedelta(days=15)).strftime("%Y-%m-%d")
        two_tests = {
            "student_id": "S123",
            "upcoming_tests": [
                {"test_id": "T1", "subject": "Math", "test_name": "Soon",
                 "date": soon,  "topics": ["TopicA"]},
                {"test_id": "T2", "subject": "Math", "test_name": "Later",
                 "date": later, "topics": ["TopicB"]},
            ],
        }
        mod = load_tools_with_fixtures(tests_data=two_tests)
        mod._profile = {**PROFILE, "weak_topics": []}  # remove weak bias
        result = mod.get_study_plan("S123")
        scores = {e["topic"]: e["priority_score"] for e in result["study_plan"]}
        self.assertGreater(scores["TopicA"], scores["TopicB"])

    def test_daily_study_time_returned(self):
        result = self.t.get_study_plan("S123")
        self.assertEqual(result["daily_study_time_minutes"], 90)

    def test_student_name_returned(self):
        result = self.t.get_study_plan("S123")
        self.assertEqual(result["student_name"], "Arjun")


# =============================================================================
# dispatch_tool
# =============================================================================

class TestDispatchTool(unittest.TestCase):

    def setUp(self):
        self.t = load_tools_with_fixtures()

    def test_known_tool_returns_json_string(self):
        result = self.t.dispatch_tool("get_weak_topics", {"student_id": "S123"})
        parsed = json.loads(result)
        self.assertIn("weak_topics", parsed)

    def test_unknown_tool_returns_error_json(self):
        result = self.t.dispatch_tool("nonexistent_tool", {})
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("nonexistent_tool", parsed["error"])

    def test_output_is_always_valid_json(self):
        result = self.t.dispatch_tool("get_upcoming_tests", {"student_id": "S123"})
        self.assertIsInstance(json.loads(result), dict)

    def test_all_four_registered_tools_are_dispatchable(self):
        calls = [
            ("get_weak_topics",    {"student_id": "S123"}),
            ("get_upcoming_tests", {"student_id": "S123"}),
            ("get_study_plan",     {"student_id": "S123"}),
            # recommend_study_material uses embeddings stub (no-op), so it's safe
            ("recommend_study_material", {"topics": ["Algebra"]}),
        ]
        for name, args in calls:
            with self.subTest(tool=name):
                result = self.t.dispatch_tool(name, args)
                self.assertIsInstance(json.loads(result), dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)