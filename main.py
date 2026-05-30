"""
main.py — FastAPI application.

Endpoints:
  GET  /           → serves the UI (static/index.html)
  POST /chat       → SSE stream of agent events
  GET  /student/{student_id} → student profile summary for UI sidebar
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent
from embeddings import build_index
from tools import _load


# ---------------------------------------------------------------------------
# Lifespan — build FAISS index once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Building embeddings index...")
    build_index()
    print("[startup] Ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Study Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    student_id: str
    query: str


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_agent(student_id: str, query: str) -> AsyncGenerator[str, None]:
    async for event in run_agent(student_id, query):
        yield _event(event)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Streams agent events as Server-Sent Events (SSE).
    Event types:
      tool_call     — agent is calling a tool
      tool_result   — tool returned data
      response_chunk — streamed text from GPT-4.1
      done          — stream complete
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return StreamingResponse(
        _stream_agent(req.student_id, req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/student/{student_id}")
async def get_student(student_id: str):
    """Returns student profile summary for the UI sidebar."""
    profile = _load("student_profile.json")
    performance = _load("performance_history.json")

    if profile["student_id"] != student_id:
        raise HTTPException(status_code=404, detail="Student not found.")

    return {
        "student_id": profile["student_id"],
        "name": profile["name"],
        "grade": profile["grade"],
        "board": profile["board"],
        "target_exam": profile["target_exam"],
        "daily_study_time_minutes": profile["daily_study_time_minutes"],
        "strong_topics": profile["strong_topics"],
        "weak_topics": profile["weak_topics"],
        "subject_performance": performance["subject_performance"],
    }


# ---------------------------------------------------------------------------
# Static files — mount last so /chat and /student take priority
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
