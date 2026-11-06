"""
RAG + tool-calling agent using Google Gemini native function calling.

Flow for each question:
  1. Retrieve top-k relevant chunks from FAISS (RAG context).
  2. Send question + retrieved context + tool declarations to Gemini.
  3. Gemini autonomously decides whether to call get_equipment_status /
     log_inspection (zero, one, or multiple times) based on the question --
     this is NOT hardcoded by question in app logic.
  4. Any tool calls are executed locally and results fed back to Gemini.
  5. Gemini produces the final answer, constrained by the system prompt to
     use ONLY the retrieved manual context for factual/manual claims, and to
     explicitly say when the manuals don't contain the answer.
  6. We return answer text + citations (doc/page) derived from the chunks
     actually retrieved, + the full tool_calls trace.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from .retriever import retrieve
from .tools import TOOL_DECLARATIONS, TOOL_IMPLS

# Load GEMINI_API_KEY from the Hermes profile .env (never hardcode the key).
load_dotenv("/home/franklyn/.hermes/profiles/kratos/.env")
load_dotenv()  # also allow a local .env to override/extend if present

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in environment. Set it in "
        "/home/franklyn/.hermes/profiles/kratos/.env or export it before running."
    )

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
TOP_K = 8

_client = genai.Client(api_key=GEMINI_API_KEY)

_FUNCTION_DECLARATIONS = [types.FunctionDeclaration(**decl) for decl in TOOL_DECLARATIONS]
_TOOL = types.Tool(function_declarations=_FUNCTION_DECLARATIONS)

SYSTEM_PROMPT = """You are a maintenance assistant for industrial pumps.

You will be given a MANUAL CONTEXT section made of retrieved excerpts from
real pump O&M manuals (each excerpt tagged with its source filename and page
number), and a maintenance question.

Rules:
1. Answer using ONLY the information in the MANUAL CONTEXT for any factual
   claim about procedures, specs, or warnings. Do NOT use outside/general
   knowledge about pumps.
2. If the MANUAL CONTEXT does not contain enough information to answer the
   question, you MUST say so explicitly (e.g. "The manuals do not contain
   information about ..."). Do not invent or guess an answer.
3. Every factual claim drawn from the manuals must be followed by a citation
   in the form (source: <filename>, page <N>).
4. You have two tools available: get_equipment_status and log_inspection.
   Decide for yourself, based on the question, whether calling a tool would
   help answer it (e.g. questions about current equipment state, alarms,
   service history, or safety-to-restart should check get_equipment_status;
   requests to log/record a finding should use log_inspection). Do not call a
   tool unless the question calls for it.
5. When a tool result and the manual procedure are both relevant (e.g. "is it
   safe to restart"), combine them: state the live equipment status/alarms
   AND the manual's relevant procedure/warning with its citation.
"""


def _build_user_message(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[{c['doc']} - page {c['page']}]\n{c['text']}")
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant chunks retrieved)"
    return f"MANUAL CONTEXT:\n{context_text}\n\nQUESTION: {question}"


def answer_question(question: str) -> Dict[str, Any]:
    chunks = retrieve(question, k=TOP_K)
    user_message = _build_user_message(question, chunks)

    contents: List[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_message)])
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[_TOOL],
    )

    tool_calls_trace: List[Dict[str, Any]] = []

    def _generate_with_retry(**kwargs):
        """Retry on transient failures with backoff:
          - Gemini 429 (rate limit) / 503 (overloaded) -- free-tier quota is
            easy to hit when running several questions back to back.
          - Network-level faults (DNS resolution failures, connect timeouts,
            dropped connections) -- a momentary resolver/network blip should
            not surface as a hard 500 to the caller.
        """
        last_exc = None
        for attempt in range(6):
            try:
                return _client.models.generate_content(**kwargs)
            except genai_errors.ClientError as e:
                last_exc = e
                if getattr(e, "code", None) in (429, 503) or "RESOURCE_EXHAUSTED" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                time.sleep(2 * (attempt + 1))
                continue
        raise last_exc

    # Agent loop: allow a few rounds of tool calling before final answer.
    max_rounds = 5
    final_text = ""
    for _ in range(max_rounds):
        response = _generate_with_retry(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts or []

        function_calls = [p for p in parts if getattr(p, "function_call", None)]
        text_parts = [p.text for p in parts if getattr(p, "text", None)]

        if not function_calls:
            final_text = "\n".join(t for t in text_parts if t)
            break

        # Echo the model's turn (including the function_call parts) back into
        # the conversation, then append the tool results as a function
        # response turn, per Gemini function-calling protocol.
        contents.append(candidate.content)

        response_parts = []
        for fc_part in function_calls:
            fc = fc_part.function_call
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}
            impl = TOOL_IMPLS.get(fn_name)
            if impl is None:
                result = {"error": f"Unknown tool '{fn_name}'"}
            else:
                try:
                    result = impl(**fn_args)
                except Exception as e:  # defensive -- never crash the request
                    result = {"error": str(e)}

            tool_calls_trace.append({"name": fn_name, "args": fn_args, "result": result})
            response_parts.append(
                types.Part.from_function_response(name=fn_name, response={"result": result})
            )

        contents.append(types.Content(role="user", parts=response_parts))
    else:
        final_text = "(Reached max tool-call rounds without a final answer.)"

    citations = [{"doc": c["doc"], "page": c["page"]} for c in chunks]

    return {
        "question": question,
        "answer": final_text,
        "citations": citations,
        "tool_calls": tool_calls_trace,
        "retrieved_chunks": chunks,  # kept for transparency/debugging
    }
