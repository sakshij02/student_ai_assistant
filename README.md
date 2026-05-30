# Study Assistant

An AI-powered personalized study assistant for CBSE students. Built with GPT-4.1, FastAPI, FAISS semantic search, and a clean chat UI.
Given a student's performance history, weak topics, and upcoming test schedule, the assistant answers natural language queries like "What should I study this week?" or "I have a Maths test coming up — help me prepare" with specific, grounded recommendations rather than generic advice.
Every response is backed by real student data retrieved via tools at query time — no hallucinated scores, dates, or material titles. The assistant reasons across weak topics, test urgency, and available study materials to produce a prioritized, actionable study plan tailored to the individual student.

## Architecture

```
User Query (UI)
      ↓
POST /chat  (FastAPI)
      ↓
GPT-4.1 with Tool Calling (agent.py)
      ↓  decides which tools to call (parallel, up to 4 rounds)
┌─────────────────────────────────────┐
│  get_weak_topics        (parallel)  │  ← student_profile + performance JSON
│  get_upcoming_tests     (parallel)  │  ← upcoming_tests JSON
│  recommend_study_material(parallel) │  ← FAISS semantic search over materials
│  get_study_plan         (parallel)  │  ← cross-references weak topics + tests
└─────────────────────────────────────┘
      ↓  tool results injected into context
GPT-4.1 generates personalized response
      ↓  streamed as SSE
Chat UI renders in real time
```


## Approach

The assistant is built around a **retrieval-augmented agentic loop**: instead of relying on the LLM's general knowledge, every response is grounded in real student data fetched via tools at query time.

**Data layer** — Student data lives in four JSON files: profile (weak/strong topics, daily study time), performance history (subject-level scores), study materials (titles mapped to topics), and upcoming tests (dates and topics covered). This is intentionally simple — the focus is on the reasoning layer, not the data store.

**Agentic loop** — `agent.py` runs a multi-round loop: GPT-4.1 decides which tools to call, all requested tools are dispatched in parallel, results are injected back into context, and the model either calls more tools or generates the final response. A safety cap of 4 rounds prevents runaway loops.

**Tool descriptions as the contract** — Orchestration logic (which tools to call together, what each tool's output is missing) lives entirely in the tool descriptions, not in the system prompt. The system prompt is limited to persona, tone, and response formatting. This keeps the system maintainable — adding a new tool means writing a good description, not editing prompt instructions.

**Retrieval** — Hybrid approach: structured filtering for profile/test data (exact JSON lookups), semantic search for materials (FAISS + OpenAI `text-embedding-3-small`, indexed at startup as `"{topic}: {title}"` strings to handle vocabulary mismatch).

**Streaming** — The `/chat` endpoint streams Server-Sent Events. Tool call events (`tool_call`, `tool_result`) are emitted before the final response so the UI can show the student what the assistant is doing in real time.

**UI** — A lightweight single-page chat interface with dark/light theme support. It displays live tool call activity (e.g. "Building priority study plan...") as the agent works, and streams the final response token by token.

## Setup

1. **Clone and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   ```

3. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

4. **Open the UI**
   Navigate to [http://localhost:8000](http://localhost:8000)

5. **Run the tests**
   ```bash
   python -m unittest discover -s tests -v
   ```

## Suggested Test Queries

- "I am weak in Algebra. What should I do next?"
- "What should I study this week?"
- "Which topic should I prioritize first?"
- "I have a Maths test coming up. Help me prepare."

## Tools

| Tool | Responsibility | Implementation |
|---|---|---|
| `get_weak_topics` | Returns weak topics with subject-level performance scores | Joins student_profile.json + performance_history.json; infers subject from topic keywords |
| `get_upcoming_tests` | Returns upcoming tests sorted by date with days remaining | Filters past tests, computes days remaining, sorts ascending |
| `recommend_study_material` | Semantic search over study materials | FAISS + OpenAI `text-embedding-3-small`; falls back to string match if index unavailable |
| `get_study_plan` | Deterministic priority ranking | Scoring: weak topic (+5), in upcoming test (+10), urgency decay (sooner test = higher bonus) |

## Project Structure

```
study-assistant/
├── main.py           # FastAPI app + SSE /chat endpoint
├── agent.py          # GPT-4.1 agentic loop with parallel tool calling
├── tools.py          # All 4 tools + OpenAI schemas + dispatcher
├── embeddings.py     # FAISS index builder + semantic search
├── requirements.txt
├── data/
│   ├── student_profile.json
│   ├── performance_history.json
│   ├── study_materials.json
│   └── upcoming_tests.json
├── static/
│   └── index.html    # Chat UI with dark/light theme
└── tests/
    ├── test_config.py    # Shared fixtures and module loaders (no test classes)
    ├── test_tools.py     # Unit tests for tools.py (40 tests)
    └── test_agent.py     # Unit tests for agent.py (14 tests)
```

## Design Decisions

- **Single-turn parallel tool calling**: GPT-4.1 decides which tools to call, dispatches them in one round, then synthesizes a response. No multi-step loop needed for this dataset.
- **Deterministic priority scoring**: `get_study_plan` uses a scoring formula so rankings are reliable and explainable, not LLM-guessed.
- **Hybrid retrieval**: Structured JSON filtering for weak topics/tests; FAISS semantic search for study materials (handles vague queries like "equations" matching "Quadratic Equations Concept Video").
- **SSE streaming**: Tool call events are streamed to the UI so the student sees what's happening in real time.

## Limitations

- **Single student only** — the data layer is hardcoded to one student profile (S123). There is no multi-student support, authentication, or user session management.
- **Static dataset** — study materials, test schedules, and performance scores are read from JSON files at startup. There is no mechanism for a student or teacher to update them without editing files directly.
- **No conversation memory** — each query is stateless. The assistant cannot refer back to what was discussed earlier in the same session.
- **Scoring formula is hand-tuned** — the +5/+10/urgency weights in `get_study_plan` are fixed constants, not learned from actual student outcomes.

## Future Improvements

- **Multi-student support** — move student data to a database (e.g. PostgreSQL) with proper authentication so the assistant can serve multiple students with isolated profiles.
- **Dynamic data ingestion** — allow teachers to upload new study materials and test schedules via an admin API; re-index FAISS automatically on update.
- **Conversation memory** — persist chat history per session and include a summary in the system prompt so the assistant can reference earlier context ("as we discussed earlier, your Algebra test is in 3 days").
- **Feedback loop** — let students mark responses as helpful or not; use that signal to tune the priority scoring weights over time.
- **Richer performance data** — incorporate topic-level scores (not just subject-level) and historical trend data (improving vs. declining) for more precise weak-area detection.
- **Adaptive study time allocation** — split the daily 90-minute budget across topics in the study plan automatically, factoring in topic difficulty and days remaining per test.