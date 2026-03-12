"""
Visualization router for Coast — Manim-powered visual explanations.
Uses GPT-4o for script generation + auto-repair,
GPT-4o-mini for concept picking / section analysis.
Mount into the main FastAPI app with: app.include_router(viz_router)
"""

import json
import os
import re
import subprocess
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

viz_router = APIRouter()

MEDIA_DIR = Path(__file__).parent / "media" / "visualizations"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# ── Client ───────────────────────────────────────────────────────────────

def _get_openai():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ── Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at writing Manim Community Edition (v0.20) Python scripts that create clear, beautiful mathematical/scientific visualizations.

RULES:
1. Always use `from manim import *`
2. Create exactly ONE Scene class called `Visualization`
3. Use self.play() for animations with appropriate run_time
4. Keep visualizations concise (under 12 seconds total)
5. Use clear colors: BLUE, YELLOW, GREEN, RED, WHITE, PURPLE, ORANGE
6. ONLY use Text() for labels — NEVER use MathTex, Tex, or LaTeX
7. For math symbols use unicode: Text("y = x²"), Text("∫ f(x) dx"), Text("Σ"), Text("Δ")
8. Use smooth animations (FadeIn, Write, Create, Transform)
9. Make the visualization self-explanatory for a student
10. DO NOT use external files or images
11. Output ONLY the Python code, no markdown fences, no explanations

TEXT LAYOUT — CRITICAL:
- Title text: font_size=28 max. Place at scene TOP with .to_edge(UP)
- Label text: font_size=20 max
- ALWAYS use buff=0.3 or more between adjacent objects via .next_to(obj, direction, buff=0.3)
- NEVER stack text without explicit positioning — every Text must have .to_edge(), .next_to(), or .move_to()
- For lists of items use VGroup(...).arrange(DOWN, buff=0.25)
- Keep all content within the Manim frame (x ∈ [-7,7], y ∈ [-4,4])

STYLE:
- Use Axes or NumberPlane for coordinate systems
- Use ValueTracker for animated parameters
- Group related objects with VGroup
- Use .animate syntax for property animations
"""

REPAIR_PROMPT = """You are a Manim debugging expert. The script below failed to render.
Fix the error and return ONLY the corrected Python code — no markdown, no explanation.

RULES (same as before):
- from manim import *
- ONE Scene class called Visualization
- Text() only, NO MathTex/Tex/LaTeX
- Unicode for math symbols
- Keep all text within frame bounds with explicit positioning
- font_size ≤ 28 for titles, ≤ 20 for labels
"""


# ── Pydantic Models ──────────────────────────────────────────────────────

class VizRequest(BaseModel):
    topic: str
    description: str

class NotebookVizRequest(BaseModel):
    notebook_id: Any = ""
    sections: list[Any] = []

class SectionVizRequest(BaseModel):
    topic: str
    content: str


# ── Core Functions ───────────────────────────────────────────────────────

def _generate_script(topic: str, description: str) -> str:
    """Generate a Manim script using GPT-4o."""
    client = _get_openai()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a Manim visualization for:\nTopic: {topic}\nDescription: {description}"},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    script = resp.choices[0].message.content.strip()
    script = re.sub(r'^```(?:python)?\n?', '', script)
    script = re.sub(r'\n?```$', '', script)
    return script


def _fix_script(script: str, error: str) -> str:
    """Feed a failed script + error back to GPT-4o for auto-repair."""
    client = _get_openai()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": REPAIR_PROMPT},
            {"role": "user", "content": f"Failed script:\n```python\n{script}\n```\n\nError:\n{error[-600:]}"},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    fixed = resp.choices[0].message.content.strip()
    fixed = re.sub(r'^```(?:python)?\n?', '', fixed)
    fixed = re.sub(r'\n?```$', '', fixed)
    return fixed


def _render_script(script: str) -> dict:
    """Render a Manim script to GIF. Returns {filename, url} or {error}."""
    viz_id = uuid.uuid4().hex[:10]
    work_dir = tempfile.mkdtemp(prefix="manim_")
    script_path = Path(work_dir) / f"scene_{viz_id}.py"
    script_path.write_text(script, encoding="utf-8")
    render_media = Path(work_dir) / "media"

    cmd = [
        "python3", "-m", "manim", "render",
        "-ql", "--format=gif",
        str(script_path), "Visualization",
        "--media_dir", str(render_media),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=work_dir)
        if result.returncode != 0:
            return {"error": f"Render failed: {(result.stderr or '')[-400:]}"}

        gifs = list(render_media.rglob("*.gif"))
        if not gifs:
            return {"error": "No output GIF found"}

        latest = max(gifs, key=lambda f: f.stat().st_mtime)
        final_name = f"viz_{viz_id}.gif"
        final_path = MEDIA_DIR / final_name
        shutil.copy2(str(latest), str(final_path))
        return {"filename": final_name, "url": f"/media/visualizations/{final_name}"}

    except subprocess.TimeoutExpired:
        return {"error": "Render timed out (60s)"}
    except Exception as e:
        return {"error": str(e)}


def _generate_and_render(topic: str, description: str, max_retries: int = 2) -> dict:
    """Generate script with GPT-4o, render it, auto-retry on failure."""
    script = _generate_script(topic, description)
    result = _render_script(script)

    attempt = 0
    while "error" in result and attempt < max_retries:
        attempt += 1
        try:
            script = _fix_script(script, result["error"])
            result = _render_script(script)
        except Exception:
            break

    return result


def _pick_concepts(sections: list[dict]) -> list[dict]:
    """Use GPT-4o-mini to pick 1-2 concepts that benefit from visualization."""
    client = _get_openai()

    section_summary = "\n".join(
        f"- {s.get('title', 'Untitled')}: {(str(s.get('content', '')) or '')[:200]}"
        for s in sections[:10]
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You pick concepts from study notes that would benefit from animated math/science visualizations.
Return a JSON array of 1-2 objects, each with:
  "topic": short concept name,
  "description": technical prompt for generating a Manim animation,
  "explanation": a student-friendly 1-2 sentence explanation of what the animation shows
Pick concepts that are VISUAL in nature (graphs, functions, geometric shapes, data distributions, biological structures, physics simulations, etc).
If no concepts are suitable for visualization, return an empty array [].
Output ONLY the JSON array, nothing else."""},
            {"role": "user", "content": f"Pick concepts for visualization from these notebook sections:\n{section_summary}"},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    text = resp.choices[0].message.content.strip()
    text = re.sub(r'^```(?:json)?\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    try:
        concepts = json.loads(text)
        if not isinstance(concepts, list):
            return []
        return concepts[:2]
    except Exception:
        return []


def _describe_section(topic: str, content: str) -> dict | None:
    """Use GPT-4o-mini to decide if a single section deserves a viz and describe it."""
    client = _get_openai()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Decide if this study section would benefit from a Manim animation.
If YES, return JSON: {"description": "<technical Manim prompt>", "explanation": "<student-friendly 1-2 sentence description>"}
If NO, return JSON: {"skip": true}
Output ONLY JSON, nothing else."""},
            {"role": "user", "content": f"Topic: {topic}\nContent:\n{content[:1000]}"},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    text = resp.choices[0].message.content.strip()
    text = re.sub(r'^```(?:json)?\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        return None


# ── Endpoints ────────────────────────────────────────────────────────────

@viz_router.get("/media/visualizations/{filename}")
async def serve_viz(filename: str):
    file_path = MEDIA_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Not found")
    media_type = "image/gif" if filename.endswith(".gif") else "image/png"
    return FileResponse(str(file_path), media_type=media_type)


@viz_router.post("/api/visualize")
async def create_viz(req: VizRequest):
    """Generate a single visualization from a topic + description."""
    result = _generate_and_render(req.topic, req.description)
    if "error" in result:
        raise HTTPException(500, detail=result["error"])
    return {"url": result["url"], "filename": result["filename"]}


@viz_router.post("/api/visualize/notebook")
async def create_notebook_viz(req: NotebookVizRequest):
    """Auto-generate visualizations for a notebook's sections."""
    raw_sections = []
    for s in req.sections:
        if isinstance(s, dict):
            raw_sections.append(s)
        else:
            raw_sections.append({"title": str(s), "content": ""})

    try:
        concepts = _pick_concepts(raw_sections)
    except Exception as e:
        raise HTTPException(500, detail=f"Concept picking failed: {e}")

    if not concepts:
        return {"visualizations": []}

    results = []
    for concept in concepts:
        topic = concept.get("topic", "")
        desc = concept.get("description", "")
        explanation = concept.get("explanation", desc)
        if not topic:
            continue
        try:
            render = _generate_and_render(topic, desc)
            if "error" not in render:
                results.append({
                    "topic": topic,
                    "description": explanation,
                    "url": render["url"],
                    "filename": render["filename"],
                })
        except Exception:
            continue

    return {"visualizations": results}


@viz_router.post("/api/visualize/section")
async def create_section_viz(req: SectionVizRequest):
    """Generate a visualization for a single notebook section."""
    spec = _describe_section(req.topic, req.content)

    if not spec or spec.get("skip"):
        return {"skip": True, "message": f"No visualization needed for '{req.topic}'."}

    desc = spec.get("description", "")
    explanation = spec.get("explanation", desc)

    if not desc:
        return {"skip": True, "message": "Could not determine a suitable visualization."}

    render = _generate_and_render(req.topic, desc)
    if "error" in render:
        return {"skip": True, "message": f"Render failed after retries: {render['error'][:200]}"}

    return {
        "topic": req.topic,
        "description": explanation,
        "url": render["url"],
        "filename": render["filename"],
    }
