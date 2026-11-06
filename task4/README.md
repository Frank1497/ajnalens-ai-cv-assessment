# Task 4 -- Maintenance Assistant with RAG and Tool Calling

A FastAPI service that answers pump maintenance questions by combining
retrieval-augmented generation (RAG) over 3 real pump O&M manuals with an
LLM that can autonomously call two tools (mock live equipment status, and
inspection logging).

> **LLM provider note (explicit decision from Franklyn):** this implementation
> uses **Google Gemini** (`google-genai` SDK, native function calling) instead
> of Anthropic. Hermes's own `ANTHROPIC_API_KEY` in the dev environment is an
> internal OAuth session token, not a standalone `sk-ant-...` Console key, and
> must never be extracted/reused by scripts. No other Anthropic key was
> available, so Gemini's native function-calling (functionally equivalent to
> Anthropic's `tool_use` blocks) is used for the LLM + tool-calling layer
> everywhere the original spec said "Claude" / "Anthropic".



## Architecture

```
task4/
  app/
    manuals/            3 source PDF pump O&M manuals
    ingest.py           PDF -> text -> chunk -> embed -> FAISS index
    retriever.py         query -> top-k relevant chunks
    tools.py             the 2 tools (+ Gemini function declarations)
    agent.py             RAG + Gemini native function-calling agent loop
    main.py              FastAPI app, POST /ask
    index/                built FAISS index + chunk metadata (generated)
  questions.json         the 5 required test questions
  outputs/
    results.json         real API responses for all 5 questions
    inspections.json     records appended by the log_inspection tool
  README.md              this file
```



## Pipeline

1. **Ingest** (`app/ingest.py`): parse each PDF page-by-page with `pypdf`,
  normalize whitespace, then chunk each page's text.
2. **Chunking choice**: `CHUNK_SIZE_WORDS = 220`, `CHUNK_OVERLAP_WORDS = 40`
  (word-count based, not raw characters).
  - **Why ~220 words**: close to the commonly-recommended 500-token chunk
  size for technical/procedural prose (English averages ~0.75 tokens per
  word, so 220 words ≈ ~290 tokens) -- big enough to keep a full
  instruction step, warning, or spec line intact, small enough that a
  retrieved chunk stays topically coherent and doesn't dilute the
  embedding with unrelated content from elsewhere on the page.
  - **Why 40-word (~18%) overlap**: pump manuals frequently put a critical
  warning or a multi-step procedure across a chunk boundary (e.g. "do not
  restart until ... " split mid-sentence). The overlap means that if a
  boundary falls mid-procedure, the full warning still appears intact in
  at least one of the two neighboring chunks, so it stays retrievable.
  - Chunking is done **within each page** (not across pages) so every chunk
  can be cited with exactly one page number -- needed for requirement #2
  (citations must include document + page).
3. **Embed**: `sentence-transformers/all-MiniLM-L6-v2` -- local, no API key,
  384-dim, fast, good general semantic-similarity quality for short
   technical passages.
4. **Vector store**: FAISS `IndexFlatIP` over L2-normalized embeddings
  (inner product on normalized vectors = cosine similarity). 171 chunks
   total across the 3 manuals (32 + 20 + 20 pages).
5. **Retrieval**: top-k=4 chunks per question, each tagged with its source
  filename and page number.
6. **LLM + tools (Gemini)**: `agent.py` sends the question + retrieved
  context + a system prompt (RAG-only answering, must cite doc+page, must
   say so explicitly when the manuals don't contain the answer) + the two
   function declarations to Gemini. Gemini decides on its own, per question,
   whether to call `get_equipment_status` and/or `log_inspection` -- this
   decision is **not hardcoded per question** in app logic; the agent loop
   just executes whatever tool call(s) Gemini requests and feeds the result
   back until Gemini produces a final text answer.
7. **FastAPI**: `POST /ask` takes `{"question": "..."}` and returns
  `{"question", "answer", "citations", "tool_calls"}`.



## Tools

- `get_equipment_status(equipment_id)` -- returns a **mock** JSON status
`{equipment_id, status, last_service_date, active_alarms}` from a small
hardcoded lookup table with deliberately varied data:
  - `P-101`: running, no alarms, serviced 2025-08-12
  - `P-102`: **stopped**, `HIGH_VIBRATION` + `SEAL_LEAK_WARNING`, serviced 2025-03-02
  - `P-103`: running, `LOW_SUCTION_PRESSURE`, serviced 2025-09-28
  - Unknown IDs return `status: "unknown"` with a note, rather than
  fabricated data.
- `log_inspection(equipment_id, note, severity)` -- appends a timestamped
record to `task4/outputs/inspections.json` (creates the file on first use).

Both are declared to Gemini as native function-calling tool declarations in
`app/tools.py` (`TOOL_DECLARATIONS`), the direct equivalent of Anthropic
`tool_use` blocks.

## Data source

The 3 pump O&M manuals were downloaded directly from each manufacturer's
public website (publicly available, free technical documentation, no login
or paywall):

- `pump_manual_centrifugal.pdf` -- Pompetravaini centrifugal pump series
(monostage/multistage/self-priming/magnetic-drive: TCH, TCA, TCT, TCD, MC,
MEC, TCK, TBA, TBAK, TMA), 32 pages.
- `pump_manual_desmi_modular.pdf` -- DESMI Pumping Technology A/S, "Modular S"
centrifugal pump series operation and maintenance instructions, 20 pages.
- `pump_manual_finishthompson_db.pdf` -- Finish Thompson Inc., DB series
sealless magnetic-drive centrifugal pumps (models DB6-DB10), 20 pages.

All 3 are manufacturer-published technical manuals, freely distributed for
end users/technicians; used here only as source text for RAG retrieval, not
redistributed or modified beyond chunking for the index. 72 pages total,
171 chunks after chunking.

## Model

`gemini-3-flash-preview` (confirmed via live API to support native function
calling in this account; `gemini-2.5-flash` / `gemini-flash-latest` were
unavailable/deprecated for this API key at the time of building -- the model
name is overridable via the `GEMINI_MODEL` env var if it changes again).
The free-tier API key used here is rate-limited to 5 requests/minute, so the
test-question runner paces requests ~15s apart; `agent.py` also has a
built-in retry/backoff for transient `429`/`503` responses.

## Setup & run

Run these commands from the **repository root** (`ai-cv-assessment`), not from inside `task4/`. The app is the package `task4.app`, so the uvicorn target is `task4.app.main:app`.

```bash
cd /home/franklyn/Documents/projects/ajnalens-assignment/ai-cv-assessment
source .venv/bin/activate
uv pip install google-genai    # one-time, if the Gemini SDK is not installed yet

# GEMINI_API_KEY: put it in the repo-root .env (see .env.example), or export it.
# agent.py also loads ~/.hermes/profiles/kratos/.env when that file exists.
#   export GEMINI_API_KEY=...

# Optional. Startup builds the FAISS index automatically when it is missing.
python -m task4.app.ingest

uvicorn task4.app.main:app --port 8000
```

The process listens on `http://127.0.0.1:8000`. Open that URL in a browser: `/` redirects to the Swagger UI at `/docs` (`GET /health`, `POST /ask`).

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}
```

Stop the server with Ctrl+C in the terminal that started it.

### curl example

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is pump P-102 safe to restart?"}' | python3 -m json.tool
```



## Test questions & results

`questions.json` has the 5 required questions. All 5 were run against the
**real running API** and saved verbatim to `outputs/results.json`:

1. "Is pump P-102 safe to restart?" -- **triggers** `get_equipment_status`;
  tool reports `stopped` + `HIGH_VIBRATION` + `SEAL_LEAK_WARNING`; answer
   combines that live status with the manual's vibration/seal/restart
   warnings (cited `pump_manual_centrifugal.pdf` p.22,
   `pump_manual_finishthompson_db.pdf` p.8) and correctly concludes **not
   safe to restart**.
2. "What is the recommended procedure before starting a centrifugal pump for
  the first time?" -- answered from the manuals only, cited.
3. "What torque setting should I use on the impeller lock nut for the Finish
  Thompson DB pump?" -- **the unanswerable question**: no tool call; the
   assistant explicitly states the manuals don't specify that exact value
   (confirmed by direct PDF inspection -- the manual only gives a torque for
   the drive/motor coupling bolts, page 10, not an "impeller lock nut").
4. "What are the current active alarms and last service date for pump
  P-103, and is it currently running?" -- **triggers** `get_equipment_status`
   for P-103 (running, `LOW_SUCTION_PRESSURE`, serviced 2025-09-28).
5. "What could cause excessive vibration or noise in a centrifugal pump, and
  what should the operator do about it?" -- answered from the manuals'
   troubleshooting/vibration-monitoring sections, cited.

`log_inspection` was separately exercised with an ad-hoc question
("log an inspection note for P-101...") and verified to append a timestamped
record to `outputs/inspections.json` -- not one of the 5 required questions,
but demonstrates the second tool also fires autonomously when the question
calls for it.

## Answers to required questions



### a) How would you measure whether retrieval is good enough?

Real, measurable approach:

- **Groundedness / faithfulness check**: have a separate LLM pass (or human)
verify every sentence in the final answer is entailed by the retrieved
chunks, catching cases where the LLM drifts into general knowledge despite
the RAG-only instruction.
- **Answer-level end-to-end eval**: for a held-out set of Q/A pairs with known
correct answers, measure exact/semantic match of the final answer, which
captures failures introduced by retrieval *and* generation.



### b) How would you extend this for "point a camera at a part, what is this and how do I replace it?"

This is a vision-language task layered on top of the existing RAG pipeline. I have built a similar capture-and-describe flow before, for an AI consulting engagement on assistive technology for blind users.

1. **Capture**: technician takes a photo (or video frame) of the part via a
  mobile app/AR glasses.
2. **Identify the part**: route the image through a vision-language model
  (e.g. Gemini's own multimodal vision input, or a dedicated CLIP-based part
   classifier fine-tuned on the equipment catalog's part photos) to produce
   a part name/ID and a short description, optionally also reading any
   visible part number/label via OCR as a cross-check.
3. **Re-use this same RAG pipeline**: feed the identified part name/ID as the
  "question" (e.g. "How do I replace the mechanical shaft seal on a DB
   series pump?") into the existing retriever + Gemini function-calling agent
   -- no new retrieval infrastructure needed, since the manuals are already
   chunked/embedded/indexed.
4. **Tool-calling extension**: add a `get_part_availability(part_id)` /
  `get_replacement_procedure_steps(part_id)` style tool so the agent can
   also check stock/spares inventory for that specific part, mirroring the
   existing `get_equipment_status` pattern.
5. **Feedback loop**: log a `log_inspection`-style record when the technician
  confirms the identification, to build a labeled dataset over time for
   fine-tuning a dedicated, cheaper part classifier instead of calling a
   general VLM for every lookup.



## AI tool assistance disclosure

This task (ingestion pipeline, RAG retriever, Gemini tool-calling agent,
FastAPI service, test running, and this README) was built with AI coding
assistance (Claude via Hermes). All code was reviewed, actually executed
against the real Gemini API and the real FAISS index, and the citations in
`outputs/results.json` were spot-checked against the source PDFs for
accuracy before being reported as final.

## Environment variable

`GEMINI_API_KEY` must be set (loaded by default from
`/home/franklyn/.hermes/profiles/kratos/.env`, or export it yourself). No
real key is committed anywhere in this repo -- see the root-level
`.env.example` for the variable name only.