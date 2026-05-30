"""
conftest.py — Shared fixtures and the tools module loader.

Imported by both test_tools.py and test_agent.py. Keep this file
free of test classes; it is infrastructure only.
"""

import importlib.util
import json
import sys
import types
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Dataset fixtures — mirror the sample dataset exactly
# ---------------------------------------------------------------------------

PROFILE = {
    "student_id": "S123",
    "name": "Arjun",
    "grade": 10,
    "board": "CBSE",
    "target_exam": "School Exams",
    "daily_study_time_minutes": 90,
    "strong_topics": ["Linear Equations", "Chemical Reactions"],
    "weak_topics": ["Algebra", "Quadratic Equations", "Light - Reflection and Refraction"],
}

PERFORMANCE = {
    "student_id": "S123",
    "subject_performance": [
        {"subject": "Mathematics", "overall_score_percentage": 52},
        {"subject": "Science",     "overall_score_percentage": 63},
    ],
}

MATERIALS = {
    "materials": [
        {"material_id": "M101", "topic": "Algebra",
         "title": "Algebra Basics Revision Notes"},
        {"material_id": "M103", "topic": "Quadratic Equations",
         "title": "Quadratic Equations Concept Video"},
        {"material_id": "M105", "topic": "Light - Reflection and Refraction",
         "title": "Ray Diagrams Explained"},
    ]
}

FUTURE_DATE = (date.today() + timedelta(days=6)).strftime("%Y-%m-%d")
PAST_DATE   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

TESTS = {
    "student_id": "S123",
    "upcoming_tests": [
        {
            "test_id": "T201",
            "subject": "Mathematics",
            "test_name": "Math Weekly Test",
            "date": FUTURE_DATE,
            "topics": ["Algebra", "Quadratic Equations"],
        }
    ],
}

TESTS_WITH_PAST = {
    "student_id": "S123",
    "upcoming_tests": [
        {
            "test_id": "T200",
            "subject": "Mathematics",
            "test_name": "Old Test",
            "date": PAST_DATE,
            "topics": ["Algebra"],
        },
        {
            "test_id": "T201",
            "subject": "Mathematics",
            "test_name": "Math Weekly Test",
            "date": FUTURE_DATE,
            "topics": ["Algebra", "Quadratic Equations"],
        },
    ],
}

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent


def load_tools_with_fixtures(tests_data=None):
    """
    Load tools.py from disk and immediately overwrite its module-level data
    globals with in-memory fixtures so no real files are touched.

    A fresh embeddings stub (returning no semantic hits) is installed each
    time so recommend_study_material falls through to _fallback_match by
    default; individual tests can override sys.modules["embeddings"] as needed.
    """
    if tests_data is None:
        tests_data = TESTS

    # Fresh embeddings stub — default: no semantic hits
    stub = types.ModuleType("embeddings")
    stub.search_materials = lambda topic, top_k=2: []
    sys.modules["embeddings"] = stub

    # Force a clean re-import of tools every call
    sys.modules.pop("tools", None)

    spec = importlib.util.spec_from_file_location("tools", ROOT / "tools.py")
    mod  = importlib.util.module_from_spec(spec)

    with patch("builtins.open", MagicMock()), \
         patch("json.load", side_effect=lambda f: {}):
        spec.loader.exec_module(mod)

    # Overwrite the globals that were set at module level during exec
    mod._profile       = PROFILE
    mod._performance   = PERFORMANCE
    mod._materials_raw = MATERIALS
    mod._tests_raw     = tests_data

    return mod


def load_agent_with_stubs(completions_side_effect, dispatch_return=None):
    """
    Load agent.py from disk with openai and tools fully stubbed out.

    Returns (agent_module, tools_stub) so callers can assert on
    tools_stub.dispatch_tool call counts / args if needed.
    """
    sys.modules.pop("agent", None)

    openai_stub = types.ModuleType("openai")
    mock_client = MagicMock()
    openai_stub.AsyncOpenAI = MagicMock(return_value=mock_client)
    sys.modules["openai"] = openai_stub

    tools_stub = types.ModuleType("tools")
    tools_stub.TOOL_SCHEMAS  = []
    tools_stub.dispatch_tool = MagicMock(
        return_value=dispatch_return or json.dumps({"result": "ok"})
    )
    tools_stub._load = lambda f: PROFILE
    sys.modules["tools"] = tools_stub

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        spec      = importlib.util.spec_from_file_location("agent", ROOT / "agent.py")
        agent_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_mod)
        agent_mod._load = lambda f: PROFILE

    from unittest.mock import AsyncMock
    mock_client.chat.completions.create = AsyncMock(side_effect=completions_side_effect)
    agent_mod._client = mock_client

    return agent_mod, tools_stub


# ---------------------------------------------------------------------------
# OpenAI response / tool-call builders (used by test_agent.py)
# ---------------------------------------------------------------------------

def make_openai_response(tool_calls=None, content=None, finish_reason=None):
    """Minimal mock of a non-streaming OpenAI chat completion response."""
    message = MagicMock()
    message.tool_calls  = tool_calls or []
    message.content     = content

    choice = MagicMock()
    choice.message      = message
    choice.finish_reason = finish_reason or ("tool_calls" if tool_calls else "stop")
    choice.delta        = message  # reused for streaming chunks

    response = MagicMock()
    response.choices = [choice]
    return response


def make_tool_call(name, arguments_dict, call_id="call_001"):
    """Minimal mock of an OpenAI tool_call object."""
    tc = MagicMock()
    tc.function.name      = name
    tc.function.arguments = json.dumps(arguments_dict)
    tc.id                 = call_id
    return tc