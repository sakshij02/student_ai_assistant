"""
tools.py — Four clearly scoped tools for the study assistant.

Each tool is responsible for exactly one thing:
  1. get_weak_topics          — what the student struggles with
  2. get_upcoming_tests       — what tests are coming and when
  3. recommend_study_material — what materials match a list of topics (semantic)
  4. get_study_plan           — priority-ranked topics based on weakness + test proximity
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data loading (loaded once at import time)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


def _load(filename: str) -> dict:
    with open(DATA_DIR / filename, "r") as f:
        return json.load(f)


_profile = _load("student_profile.json")
_performance = _load("performance_history.json")
_materials_raw = _load("study_materials.json")
_tests_raw = _load("upcoming_tests.json")


# ---------------------------------------------------------------------------
# Tool 1: get_weak_topics
# ---------------------------------------------------------------------------

def get_weak_topics(student_id: str) -> dict[str, Any]:
    """
    Returns the student's weak topics enriched with subject-level performance scores.
    Combines student_profile.json (weak topic list) with performance_history.json (scores).
    """
    if _profile["student_id"] != student_id:
        return {"error": f"No student found with id {student_id}"}

    # Build a subject score lookup
    score_by_subject: dict[str, float] = {
        entry["subject"]: entry["overall_score_percentage"]
        for entry in _performance["subject_performance"]
    }

    # Map weak topics to their subject (heuristic: Math topics → Mathematics, rest → Science)
    math_keywords = {"algebra", "equation", "linear", "quadratic", "geometry", "trigonometry"}

    def infer_subject(topic: str) -> str:
        if any(kw in topic.lower() for kw in math_keywords):
            return "Mathematics"
        return "Science"

    weak_topics_enriched = []
    for topic in _profile["weak_topics"]:
        subject = infer_subject(topic)
        weak_topics_enriched.append({
            "topic": topic,
            "subject": subject,
            "subject_score_percentage": score_by_subject.get(subject, None),
        })

    return {
        "student_name": _profile["name"],
        "weak_topics": weak_topics_enriched,
        "strong_topics": _profile["strong_topics"],
        "daily_study_time_minutes": _profile["daily_study_time_minutes"],
    }


# ---------------------------------------------------------------------------
# Tool 2: get_upcoming_tests
# ---------------------------------------------------------------------------

def get_upcoming_tests(student_id: str) -> dict[str, Any]:
    """
    Returns upcoming tests sorted by date (soonest first).
    Includes days_remaining so the agent can reason about urgency.
    """
    if _tests_raw["student_id"] != student_id:
        return {"error": f"No tests found for student id {student_id}"}

    today = date.today()
    upcoming = []

    for test in _tests_raw["upcoming_tests"]:
        test_date = datetime.strptime(test["date"], "%Y-%m-%d").date()
        days_remaining = (test_date - today).days
        if days_remaining >= 0:
            upcoming.append({
                "test_id": test["test_id"],
                "test_name": test["test_name"],
                "subject": test["subject"],
                "date": test["date"],
                "days_remaining": days_remaining,
                "topics": test["topics"],
            })

    # Sort by soonest first
    upcoming.sort(key=lambda t: t["days_remaining"])

    return {
        "upcoming_tests": upcoming,
        "total_count": len(upcoming),
    }


# ---------------------------------------------------------------------------
# Tool 3: recommend_study_material
# ---------------------------------------------------------------------------

def recommend_study_material(topics: list[str]) -> dict[str, Any]:
    """
    Given a list of topics, returns the best matching study material per topic.
    Uses semantic search via FAISS embeddings (see embeddings.py).
    Falls back to exact/partial string match if embeddings are unavailable.
    """
    from embeddings import search_materials  # lazy import to avoid circular deps

    results: dict[str, Any] = {}

    for topic in topics:
        matches = search_materials(topic, top_k=2)
        results[topic] = matches if matches else _fallback_match(topic)

    return {"recommendations": results}


def _fallback_match(topic: str) -> list[dict]:
    """Exact/partial string match fallback if FAISS is not ready."""
    topic_lower = topic.lower()
    matched = [
        m for m in _materials_raw["materials"]
        if topic_lower in m["topic"].lower() or m["topic"].lower() in topic_lower
    ]
    return matched[:2] if matched else []


# ---------------------------------------------------------------------------
# Tool 4: get_study_plan
# ---------------------------------------------------------------------------

def get_study_plan(student_id: str) -> dict[str, Any]:
    """
    Cross-references weak topics with upcoming tests to produce a
    deterministic priority-ranked study plan.

    Scoring:
      +10 if topic appears in an upcoming test
      +5  if topic is in the student's weak_topics list
      Bonus: -0.1 * days_remaining (sooner test = higher urgency)

    The LLM should use this ranked list to narrate a concrete study plan.
    """
    if _profile["student_id"] != student_id:
        return {"error": f"No student found with id {student_id}"}

    today = date.today()
    weak_topics: set[str] = set(_profile["weak_topics"])

    # Collect all topics that appear in upcoming tests
    test_topic_map: dict[str, list[dict]] = {}  # topic → list of tests it appears in
    for test in _tests_raw["upcoming_tests"]:
        test_date = datetime.strptime(test["date"], "%Y-%m-%d").date()
        days_remaining = (test_date - today).days
        if days_remaining < 0:
            continue
        for topic in test["topics"]:
            if topic not in test_topic_map:
                test_topic_map[topic] = []
            test_topic_map[topic].append({
                "test_name": test["test_name"],
                "date": test["date"],
                "days_remaining": days_remaining,
            })

    # Union of weak topics + test topics to consider
    all_topics = weak_topics | set(test_topic_map.keys())

    ranked = []
    for topic in all_topics:
        score = 0.0
        in_upcoming_test = topic in test_topic_map
        is_weak = topic in weak_topics

        if in_upcoming_test:
            score += 10
            # Urgency bonus: closer test = higher priority
            min_days = min(t["days_remaining"] for t in test_topic_map[topic])
            score += max(0, 10 - min_days * 0.5)  # decays with days remaining

        if is_weak:
            score += 5

        ranked.append({
            "topic": topic,
            "priority_score": round(score, 2),
            "is_weak": is_weak,
            "in_upcoming_test": in_upcoming_test,
            "upcoming_tests": test_topic_map.get(topic, []),
        })

    ranked.sort(key=lambda x: x["priority_score"], reverse=True)

    return {
        "study_plan": ranked,
        "daily_study_time_minutes": _profile["daily_study_time_minutes"],
        "student_name": _profile["name"],
    }


# ---------------------------------------------------------------------------
# OpenAI tool schemas (used by agent.py)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weak_topics",
            "description": (
                "Returns the student's weak topics, each enriched with its subject name and "
                "that subject's overall score percentage. Also returns strong topics and daily "
                "study time. Source: student_profile.json + performance_history.json."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "The student's unique ID, e.g. S123",
                    }
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_tests",
            "description": (
                "Returns all future tests for the student, sorted by date ascending. "
                "Each entry includes test name, subject, date, days remaining until the test, "
                "and the list of topics the test covers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "The student's unique ID, e.g. S123",
                    }
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_study_material",
            "description": (
                "The only tool that returns actual study resource titles (notes, videos, etc.). "
                "Accepts a list of topic names and returns the top matching materials per topic "
                "via semantic search (FAISS embeddings). Each result includes material_id, topic, "
                "and title. Returns up to 2 materials per topic. Call this whenever the response "
                "should include specific resources the student can open and study."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of topic names to retrieve study materials for.",
                    }
                },
                "required": ["topics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_study_plan",
            "description": (
                "Returns a priority-ranked list of topic names the student should study — "
                "topics and scores only, no study materials or resource titles. Scores each "
                "topic by combining weakness signal (+5 if in weak_topics) and test urgency "
                "(+10 if in an upcoming test, with a time-decay bonus for sooner tests). Each "
                "entry includes topic, priority_score, is_weak flag, and which upcoming tests "
                "cover it. Does not return any materials — pass the top topic names from this "
                "result to recommend_study_material to get actual resources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "The student's unique ID, e.g. S123",
                    }
                },
                "required": ["student_id"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher (called by agent.py)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_weak_topics": get_weak_topics,
    "get_upcoming_tests": get_upcoming_tests,
    "recommend_study_material": recommend_study_material,
    "get_study_plan": get_study_plan,
}


def dispatch_tool(name: str, arguments: dict) -> str:
    """Calls the named tool and returns its result as a JSON string."""
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = TOOL_REGISTRY[name](**arguments)
    return json.dumps(result, indent=2)