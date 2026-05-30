"""
test_agent.py — Unit tests for agent.py.

Coverage:
  _build_system_prompt — contains name, grade, board, target exam, study time,
                         student ID; contains no tool orchestration directives
  run_agent            — no tool calls → chunks + done, no tool events
                         one tool round → tool_call + tool_result + response + done
                         tool_call event carries the correct label
                         dispatch_tool called with the right args
                         max-rounds cap fires at 4 and still ends with done
                         unknown tool name surfaces as error in tool_result
                         unlisted tool gets the generic fallback label

Run with:
    python -m unittest test_agent -v
"""

import json
import unittest

from test_config import (
    PROFILE,
    load_agent_with_stubs,
    make_openai_response,
    make_tool_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect(gen):
    events = []
    async for event in gen:
        events.append(event)
    return events


def _chunk(text="x"):
    """Return a minimal streaming chunk mock."""
    from unittest.mock import MagicMock
    c = MagicMock()
    c.choices[0].delta.content = text
    return c


async def _fake_stream(*texts):
    for text in texts:
        yield _chunk(text)


# =============================================================================
# _build_system_prompt
# =============================================================================

class TestBuildSystemPrompt(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # load_agent_with_stubs needs at least one side effect; give it a
        # no-tool response + a one-chunk stream so the module loads cleanly.
        self.agent, _ = load_agent_with_stubs(
            completions_side_effect=[
                make_openai_response(tool_calls=[], finish_reason="stop"),
                _fake_stream("ok"),
            ]
        )

    def _prompt(self):
        from unittest.mock import patch
        with patch.object(self.agent, "_load", return_value=PROFILE):
            return self.agent._build_system_prompt("S123")

    def test_contains_student_name(self):
        self.assertIn("Arjun", self._prompt())

    def test_contains_grade(self):
        self.assertIn("10", self._prompt())

    def test_contains_board(self):
        self.assertIn("CBSE", self._prompt())

    def test_contains_target_exam(self):
        self.assertIn("School Exams", self._prompt())

    def test_contains_daily_study_time(self):
        self.assertIn("90", self._prompt())

    def test_contains_student_id(self):
        self.assertIn("S123", self._prompt())

    def test_no_tool_orchestration_directives(self):
        """Orchestration logic must live in tool descriptions, not the system prompt."""
        prompt = self._prompt()
        forbidden = ["always call BOTH", "never skip", "MUST call", "tool_choice"]
        for phrase in forbidden:
            self.assertNotIn(
                phrase.lower(), prompt.lower(),
                msg=f"System prompt should not contain orchestration rule: '{phrase}'",
            )


# =============================================================================
# run_agent — agentic loop behaviour
# =============================================================================

class TestRunAgent(unittest.IsolatedAsyncioTestCase):

    # ── no tool calls → immediate final response ──────────────────────────────

    async def test_no_tool_calls_yields_chunks_and_done(self):
        agent, _ = load_agent_with_stubs([
            make_openai_response(tool_calls=[], finish_reason="stop"),
            _fake_stream("Hello", " Arjun"),
        ])
        events     = await _collect(agent.run_agent("S123", "Hello"))
        types_seen = {e["type"] for e in events}
        self.assertIn("response_chunk", types_seen)
        self.assertIn("done", types_seen)
        self.assertNotIn("tool_call", types_seen)

    # ── one round of tool calls ───────────────────────────────────────────────

    async def test_tool_round_yields_all_four_event_types(self):
        tc = make_tool_call("get_weak_topics", {"student_id": "S123"})
        agent, _ = load_agent_with_stubs([
            make_openai_response(tool_calls=[tc], finish_reason="tool_calls"),
            make_openai_response(tool_calls=[],   finish_reason="stop"),
            _fake_stream("Done"),
        ])
        events     = await _collect(agent.run_agent("S123", "What are my weak topics?"))
        types_seen = {e["type"] for e in events}
        self.assertIn("tool_call",      types_seen)
        self.assertIn("tool_result",    types_seen)
        self.assertIn("response_chunk", types_seen)
        self.assertIn("done",           types_seen)

    async def test_tool_call_event_carries_correct_label(self):
        tc = make_tool_call("get_weak_topics", {"student_id": "S123"})
        agent, _ = load_agent_with_stubs([
            make_openai_response(tool_calls=[tc]),
            make_openai_response(tool_calls=[]),
            _fake_stream("x"),
        ])
        events     = await _collect(agent.run_agent("S123", "q"))
        tool_event = next(e for e in events if e["type"] == "tool_call")
        self.assertEqual(tool_event["label"], "Fetching weak topics...")

    async def test_dispatch_tool_called_with_correct_name_and_args(self):
        tc = make_tool_call("get_study_plan", {"student_id": "S123"}, call_id="c1")
        agent, tools_stub = load_agent_with_stubs([
            make_openai_response(tool_calls=[tc]),
            make_openai_response(tool_calls=[]),
            _fake_stream("x"),
        ])
        await _collect(agent.run_agent("S123", "plan?"))
        tools_stub.dispatch_tool.assert_called_once_with(
            "get_study_plan", {"student_id": "S123"}
        )

    # ── max-rounds safety cap ─────────────────────────────────────────────────

    async def test_agent_stops_after_4_rounds_and_still_emits_done(self):
        tc           = make_tool_call("get_weak_topics", {"student_id": "S123"})
        always_tools = make_openai_response(tool_calls=[tc])

        # 4 tool rounds + 1 final streaming call
        agent, _ = load_agent_with_stubs(
            [always_tools] * 4 + [_fake_stream("forced")]
        )
        events = await _collect(agent.run_agent("S123", "q"))

        self.assertEqual(events[-1]["type"], "done")
        # 4 agentic rounds + 1 final streaming create() = 5 total
        self.assertEqual(agent._client.chat.completions.create.call_count, 5)

    # ── unknown tool ──────────────────────────────────────────────────────────

    async def test_unknown_tool_surfaces_error_in_tool_result_event(self):
        tc = make_tool_call("does_not_exist", {})
        agent, _ = load_agent_with_stubs(
            [
                make_openai_response(tool_calls=[tc]),
                make_openai_response(tool_calls=[]),
                _fake_stream("x"),
            ],
            dispatch_return=json.dumps({"error": "Unknown tool: does_not_exist"}),
        )
        events       = await _collect(agent.run_agent("S123", "q"))
        result_event = next(e for e in events if e["type"] == "tool_result")
        self.assertIn("error", result_event["result"])

    # ── fallback label ────────────────────────────────────────────────────────

    async def test_unlisted_tool_gets_generic_running_label(self):
        tc = make_tool_call("some_new_tool", {"student_id": "S123"})
        agent, _ = load_agent_with_stubs([
            make_openai_response(tool_calls=[tc]),
            make_openai_response(tool_calls=[]),
            _fake_stream("x"),
        ])
        events     = await _collect(agent.run_agent("S123", "q"))
        tool_event = next(e for e in events if e["type"] == "tool_call")
        self.assertEqual(tool_event["label"], "Running some_new_tool...")


if __name__ == "__main__":
    unittest.main(verbosity=2)