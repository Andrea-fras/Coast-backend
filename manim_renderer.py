"""
Manim visualization renderer for Coast.

Uses GPT to generate Manim CE scripts from concept descriptions,
renders them to GIF/PNG, and returns the result.
"""

import os
import subprocess
import tempfile
import uuid
import re
from pathlib import Path

from openai import OpenAI

MEDIA_DIR = Path(__file__).parent / "media" / "visualizations"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an expert at writing Manim Community Edition (v0.20) Python scripts that create clear, beautiful mathematical/scientific visualizations.

RULES:
1. Always use `from manim import *`
2. Create exactly ONE Scene class called `Visualization`
3. Use self.play() for animations with appropriate run_time
4. Keep visualizations concise (under 15 seconds total)
5. Use clear colors: BLUE, YELLOW, GREEN, RED, WHITE, PURPLE, ORANGE
6. Add labels and text using MathTex or Text for clarity
7. Use smooth animations (FadeIn, Write, Create, Transform, etc.)
8. Set background to transparent or dark (default is fine)
9. Make the visualization self-explanatory — a student should understand the concept just by watching
10. DO NOT use any external files or images
11. Output ONLY the Python code, no explanations

STYLE GUIDELINES:
- Use NumberPlane or Axes for coordinate systems
- Use ValueTracker for animated parameters
- Prefer MathTex over Tex for math expressions
- Group related objects with VGroup
- Use .animate syntax for property animations
"""

USER_TEMPLATE = """Create a Manim visualization for this concept:

Topic: {topic}
Description: {description}

Generate a clear, educational animation that helps a student understand this concept visually."""


def generate_manim_script(topic: str, description: str) -> str:
    """Use GPT to generate a Manim script for the given concept."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(topic=topic, description=description)},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    script = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    script = re.sub(r'^```(?:python)?\n?', '', script)
    script = re.sub(r'\n?```$', '', script)
    return script


def render_visualization(script: str, output_format: str = "gif") -> dict:
    """
    Render a Manim script to a GIF or PNG.
    Returns {"path": str, "filename": str} on success,
    or {"error": str} on failure.
    """
    viz_id = uuid.uuid4().hex[:10]
    work_dir = tempfile.mkdtemp(prefix="manim_")
    script_path = Path(work_dir) / f"scene_{viz_id}.py"

    script_path.write_text(script, encoding="utf-8")

    if output_format == "gif":
        quality_flag = "-ql"  # low quality for speed
        fmt_flag = "--format=gif"
    else:
        quality_flag = "-ql"
        fmt_flag = "--format=png"
        # For PNG we render a single frame (last frame)

    cmd = [
        "python3", "-m", "manim", "render",
        quality_flag,
        fmt_flag,
        str(script_path),
        "Visualization",
        "--media_dir", str(MEDIA_DIR),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=work_dir,
        )

        if result.returncode != 0:
            return {
                "error": f"Manim render failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}"
            }

        # Find the output file
        if output_format == "gif":
            pattern = "**/*.gif"
        else:
            pattern = "**/*.png"

        output_files = list(MEDIA_DIR.rglob(pattern[-5:]))
        if not output_files:
            return {"error": "Render succeeded but no output file found"}

        latest = max(output_files, key=lambda f: f.stat().st_mtime)

        final_name = f"viz_{viz_id}.{output_format}"
        final_path = MEDIA_DIR / final_name
        latest.rename(final_path)

        return {
            "path": str(final_path),
            "filename": final_name,
        }

    except subprocess.TimeoutExpired:
        return {"error": "Render timed out (60s limit)"}
    except Exception as e:
        return {"error": str(e)}


DEMO_SCRIPTS = {
    "quadratic": '''from manim import *

class Visualization(Scene):
    def construct(self):
        axes = Axes(x_range=[-4, 4, 1], y_range=[-2, 8, 1], x_length=7, y_length=5)
        graph = axes.plot(lambda x: x**2, color=YELLOW, x_range=[-3, 3])
        title = Text("y = x\\u00b2", font_size=36, color=YELLOW).to_corner(UL)
        vertex = Dot(axes.c2p(0, 0), color=RED, radius=0.1)
        vlabel = Text("Vertex", font_size=20, color=RED).next_to(vertex, DOWN)

        self.play(Create(axes), run_time=1)
        self.play(Create(graph), FadeIn(title), run_time=1.5)
        self.play(FadeIn(vertex), Write(vlabel), run_time=1)
        self.wait(0.5)

        tracker = ValueTracker(1)
        moving = always_redraw(lambda: axes.plot(lambda x: tracker.get_value() * x**2, color=BLUE, x_range=[-3, 3]))
        coeff = always_redraw(lambda: Text(f"a = {tracker.get_value():.1f}", font_size=28, color=BLUE).to_corner(UR))

        self.play(Transform(graph, moving), FadeIn(coeff), run_time=0.8)
        self.remove(graph)
        self.add(moving)
        self.play(tracker.animate.set_value(2), run_time=1.5)
        self.play(tracker.animate.set_value(0.3), run_time=1.5)
        self.play(tracker.animate.set_value(-1), run_time=1.5)
        self.play(tracker.animate.set_value(1), run_time=1)
        self.wait(0.5)
''',
    "sine": '''from manim import *

class Visualization(Scene):
    def construct(self):
        axes = Axes(x_range=[-1, 7, 1], y_range=[-2, 2, 1], x_length=8, y_length=4)
        sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE, x_range=[0, 2*PI])
        cos_graph = axes.plot(lambda x: np.cos(x), color=YELLOW, x_range=[0, 2*PI])
        sin_lbl = Text("sin(x)", font_size=24, color=BLUE).next_to(axes.c2p(PI/2, 1), UP)
        cos_lbl = Text("cos(x)", font_size=24, color=YELLOW).next_to(axes.c2p(0, 1), UP)

        self.play(Create(axes), run_time=1)
        self.play(Create(sin_graph), Write(sin_lbl), run_time=2)
        self.wait(0.5)
        self.play(Create(cos_graph), Write(cos_lbl), run_time=2)

        tracker = ValueTracker(1)
        amp_graph = always_redraw(lambda: axes.plot(lambda x: tracker.get_value() * np.sin(x), color=GREEN, x_range=[0, 2*PI]))
        amp_text = always_redraw(lambda: Text(f"A = {tracker.get_value():.1f}", font_size=28, color=GREEN).to_corner(UR))
        self.play(FadeIn(amp_graph), FadeIn(amp_text), run_time=0.5)
        self.play(tracker.animate.set_value(2), run_time=1.5)
        self.play(tracker.animate.set_value(0.5), run_time=1.5)
        self.play(tracker.animate.set_value(1), run_time=1)
        self.wait(0.5)
''',
    "linear": '''from manim import *

class Visualization(Scene):
    def construct(self):
        axes = Axes(x_range=[-5, 5, 1], y_range=[-5, 5, 1], x_length=7, y_length=7, axis_config={"include_numbers": False})
        title = Text("Linear Equations: y = mx + b", font_size=30, color=WHITE).to_edge(UP)
        self.play(Create(axes), Write(title), run_time=1.5)

        tracker_m = ValueTracker(1)
        tracker_b = ValueTracker(0)
        line = always_redraw(lambda: axes.plot(lambda x: tracker_m.get_value() * x + tracker_b.get_value(), color=YELLOW, x_range=[-5, 5]))
        info = always_redraw(lambda: Text(f"m = {tracker_m.get_value():.1f}  b = {tracker_b.get_value():.1f}", font_size=24, color=YELLOW).to_corner(UR))

        self.play(Create(line), FadeIn(info), run_time=1)
        self.play(tracker_m.animate.set_value(3), run_time=1.5)
        self.play(tracker_m.animate.set_value(-1), run_time=1.5)
        self.play(tracker_b.animate.set_value(2), run_time=1.5)
        self.play(tracker_b.animate.set_value(-2), run_time=1.5)
        self.play(tracker_m.animate.set_value(1), tracker_b.animate.set_value(0), run_time=1)
        self.wait(0.5)
''',
    "cell": '''from manim import *

class Visualization(Scene):
    def construct(self):
        title = Text("Animal Cell Structure", font_size=32, color=WHITE).to_edge(UP)
        self.play(Write(title), run_time=1)

        cell = Circle(radius=2.5, color=BLUE, fill_opacity=0.15, stroke_width=3)
        cell_label = Text("Cell Membrane", font_size=16, color=BLUE).next_to(cell, DOWN, buff=0.2)
        self.play(Create(cell), Write(cell_label), run_time=1)

        nucleus = Circle(radius=0.8, color=PURPLE, fill_opacity=0.3, stroke_width=2).shift(LEFT*0.3)
        nuc_label = Text("Nucleus", font_size=14, color=PURPLE).next_to(nucleus, DOWN, buff=0.1)
        self.play(Create(nucleus), Write(nuc_label), run_time=1)

        mito = Ellipse(width=1.2, height=0.5, color=GREEN, fill_opacity=0.3).shift(RIGHT*1.3 + UP*0.8)
        mito_label = Text("Mitochondria", font_size=12, color=GREEN).next_to(mito, RIGHT, buff=0.1)
        self.play(Create(mito), Write(mito_label), run_time=0.8)

        er_pts = [UP*0.3+RIGHT*0.5, UP*0.8+RIGHT*0.8, UP*1.2+RIGHT*0.3, UP*1.5+RIGHT*0.7]
        er = VMobject(color=ORANGE, stroke_width=2)
        er.set_points_smoothly([cell.get_center() + p for p in er_pts])
        er_label = Text("ER", font_size=12, color=ORANGE).next_to(er, RIGHT, buff=0.1)
        self.play(Create(er), Write(er_label), run_time=0.8)

        ribo = VGroup(*[Dot(cell.get_center() + d, radius=0.05, color=YELLOW) for d in [RIGHT*1.8+DOWN*0.5, RIGHT*1.5+DOWN*1, LEFT*1.5+UP*1, LEFT*1+DOWN*1.2, RIGHT*0.5+DOWN*1.5]])
        ribo_label = Text("Ribosomes", font_size=12, color=YELLOW).next_to(ribo[-1], DOWN, buff=0.1)
        self.play(FadeIn(ribo), Write(ribo_label), run_time=0.8)

        self.wait(1.5)
''',
}


def create_visualization(topic: str, description: str, output_format: str = "gif") -> dict:
    """
    End-to-end: generate script from concept, render it, return result.
    Returns {"path", "filename", "script"} on success, {"error"} on failure.
    """
    try:
        script = generate_manim_script(topic, description)
    except Exception as e:
        # Fallback to demo scripts if API fails
        script = _pick_demo_script(topic, description)
        if not script:
            return {"error": f"Script generation failed (API: {e}). No demo fallback available."}

    result = render_visualization(script, output_format)
    if "error" not in result:
        result["script"] = script
    return result


def create_demo_visualization(demo_key: str, output_format: str = "gif") -> dict:
    """Render a pre-built demo visualization by key."""
    script = DEMO_SCRIPTS.get(demo_key)
    if not script:
        return {"error": f"Unknown demo: {demo_key}. Available: {list(DEMO_SCRIPTS.keys())}"}
    result = render_visualization(script, output_format)
    if "error" not in result:
        result["script"] = script
    return result


def _pick_demo_script(topic: str, description: str) -> str | None:
    """Try to match a demo script based on keywords."""
    text = (topic + " " + description).lower()
    if any(w in text for w in ["quadratic", "parabola", "x²", "x^2"]):
        return DEMO_SCRIPTS["quadratic"]
    if any(w in text for w in ["sine", "cosine", "trig", "sin", "cos", "wave"]):
        return DEMO_SCRIPTS["sine"]
    if any(w in text for w in ["linear", "slope", "y=mx", "straight line"]):
        return DEMO_SCRIPTS["linear"]
    if any(w in text for w in ["cell", "biology", "mitochondria", "nucleus", "membrane"]):
        return DEMO_SCRIPTS["cell"]
    return DEMO_SCRIPTS.get("quadratic")
