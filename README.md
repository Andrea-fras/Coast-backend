# Coast backend

FastAPI service behind app.coast.academy: accounts, lecture uploads and
indexing, Pedro (lessons, chat, Ask sources), notes, the exploration map and
treasure chests. Student and course memory live in OMA (`coast_content_oma/`).

## Setup

```bash
pip install -r requirements.txt
# PDF and OCR tools
brew install poppler tesseract                  # macOS
apt-get install poppler-utils tesseract-ocr     # Debian/Ubuntu
```

Create `.env` with the provider keys you use: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, plus `PEDRO_PROVIDER`, `RAG_PROVIDER` and
`STUDENT_OMA_ENABLED`. Production also needs `JWT_SECRET` (see `render.yaml`).

## Run

```bash
python3 server.py          # http://localhost:8000
```

Data lives next to the code locally (`coast.db`, `oma.db`, `chroma_data/`)
and on the persistent disk at `/data` in production.

## Tests

```bash
cd scripts && python3 test_http_integrity.py   # and the other test_*.py files
```

## Map terrain

`map_terrain_types.json` (The Lumen Reaches) and `map_terrain_types_l2.json`
(Neon Meridian) are exported from the app's world generators and must be
re-exported whenever those change:

```bash
node ../Coast/testing/scripts/export-map-terrain.mjs map_terrain_types.json
node ../Coast/testing/scripts/export-map-terrain.mjs map_terrain_types_l2.json --level 2
```

## Deploy

Render (`render.yaml`): `gunicorn server:app` with a single Uvicorn worker.
