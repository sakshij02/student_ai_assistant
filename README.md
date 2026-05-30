# Study Assistant

An AI-powered personalized study assistant for CBSE students. Built with GPT-4.1, FastAPI, FAISS semantic search, and a clean chat UI.

## Architecture

```
User Query (UI)
      ↓
POST /chat  (FastAPI)
      ↓
GPT-4.1 with Tool Calling (agent.py)
      ↓  decides which tools to call
┌─────────────────────────────────────┐
│  get_weak_topics                    │  ← student_profile + performance JSON
│  get_upcoming_tests                 │  ← upcoming_tests JSON
│  recommend_study_material           │  ← FAISS semantic search over materials
│  get_study_plan                     │  ← cross-references weak topics + tests
└─────────────────────────────────────┘
      ↓  tool results injected into context
GPT-4.1 generates personalized response
      ↓  streamed as SSE
Chat UI renders in real time
```

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
   Run all tests:
   ```bash
   python -m unittest discover -s tests -v
   ```

## Suggested Test Queries

- "I am weak in Algebra. What should I do next?"
- "What should I study this week?"
- "Which topic should I prioritize first?"
- "I have a Maths test coming up. Help me prepare."

## Tools

| Tool | Responsibility |
|---|---|
| `get_weak_topics` | Returns weak topics with subject-level performance scores |
| `get_upcoming_tests` | Returns upcoming tests sorted by date with days remaining |
| `recommend_study_material` | Semantic search (FAISS + OpenAI embeddings) over study materials |
| `get_study_plan` | Deterministic priority ranking: weak topic (+5) + upcoming test (+10) + urgency bonus |

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