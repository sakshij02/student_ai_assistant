"""
agent.py — GPT-4.1 agentic loop with parallel tool calling.

Flow:
  1. Build system prompt (lean — tools are the source of truth, not the prompt)
  2. Send user query + tool schemas to GPT-4.1
  3. Dispatch all tool calls the model requests (Round 1)
  4. Feed results back; if model requests more tools, dispatch again (Round 2)
  5. Stream the final personalized response
"""

import json
import logging
import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from tools import TOOL_SCHEMAS, dispatch_tool, _load

# ---------------------------------------------------------------------------
# Logging — structured, visible in uvicorn output
# ---------------------------------------------------------------------------

logger = logging.getLogger("study_assistant.agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(student_id: str) -> str:
    profile = _load("student_profile.json")

    return f"""You are a personalized AI study assistant for {profile['name']}, \
a Grade {profile['grade']} {profile['board']} student preparing for {profile['target_exam']}.

Response guidelines:
- Be specific and actionable — always cite actual material titles and test names from tool results.
- Factor in the student's daily study time of {profile['daily_study_time_minutes']} minutes when suggesting how to split their time.
- Flag tests within 7 days as urgent.
- Keep the tone warm and encouraging.
- Never fabricate scores, dates, topics, or material titles — all of that comes from tools.

Student ID: {student_id}"""


# ---------------------------------------------------------------------------
# Human-readable tool call labels (for UI streaming)
# ---------------------------------------------------------------------------

TOOL_LABELS = {
    "get_weak_topics": "Fetching weak topics...",
    "get_upcoming_tests": "Checking upcoming tests...",
    "recommend_study_material": "Finding study materials...",
    "get_study_plan": "Building priority study plan...",
}

# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

async def run_agent(
    student_id: str,
    user_query: str,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields SSE event dicts:
      {"type": "tool_call",      "label": "...", "tool": "..."}
      {"type": "tool_result",    "tool": "...",  "result": {...}}
      {"type": "response_chunk", "text": "..."}
      {"type": "done"}
    """
    logger.info("=" * 60)
    logger.info(f"New query | student={student_id} | query='{user_query}'")

    system_prompt = _build_system_prompt(student_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]

    # ── Agentic loop: keep calling tools until the model stops requesting them ──
    round_num = 0
    while True:
        round_num += 1
        logger.info(f"[Round {round_num}] Calling GPT-4.1 | messages in context: {len(messages)}")

        response = await _client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.3,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        logger.info(
            f"[Round {round_num}] finish_reason={finish_reason} | "
            f"tool_calls={[tc.function.name for tc in (message.tool_calls or [])]}"
        )

        # No tool calls → model is ready to give the final answer
        if not message.tool_calls:
            logger.info(f"[Round {round_num}] No tool calls — streaming final response")
            break

        # Append assistant message with tool_calls before dispatching
        messages.append(message)

        # ── Dispatch every tool the model requested ──
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            raw_args  = tool_call.function.arguments
            arguments = json.loads(raw_args)
            label     = TOOL_LABELS.get(tool_name, f"Running {tool_name}...")

            logger.info(f"  → Tool: {tool_name} | args: {arguments}")
            yield {"type": "tool_call", "label": label, "tool": tool_name}

            result_str  = dispatch_tool(tool_name, arguments)
            result_data = json.loads(result_str)

            logger.info(f"  ← Result [{tool_name}]: {result_str[:300]}{'...' if len(result_str) > 300 else ''}")
            yield {"type": "tool_result", "tool": tool_name, "result": result_data}

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      result_str,
            })

        # Safety cap: stop after 4 rounds to prevent runaway loops
        if round_num >= 4:
            logger.warning("Reached max tool-calling rounds (4) — forcing final response")
            break

    # ── Stream the final response ──
    logger.info("Streaming final response to client")

    final_response = await _client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.4,
        stream=True,
    )

    async for chunk in final_response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield {"type": "response_chunk", "text": delta.content}

    logger.info("Response complete")
    yield {"type": "done"}