"""Past paper PDF → tagged JSON pipeline.

Admin-only: scans PDFs in past_papers/{course}/ directories,
extracts questions via LLM, auto-tags them, and saves as JSON
in the scanned_papers/ output directory.

Usage:
  python paper_scanner.py                    # scan all courses
  python paper_scanner.py --course QM1       # scan one course
  python paper_scanner.py --file paper.pdf --course QM1  # scan one file
"""

from __future__ import annotations

import json
import os
import re
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAST_PAPERS_DIR = Path(__file__).parent / "past_papers"
SCANNED_OUTPUT_DIR = Path(__file__).parent / "scanned_papers"
SCANNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTRACT_SYSTEM_PROMPT = """\
You are an expert exam paper parser. Given the raw text extracted from a past exam paper PDF, \
extract ALL questions into a structured JSON format.

For EACH question, determine:
1. Whether it is "multiple-choice" or "open-ended"
2. The full question text (preserve any mathematical notation)
3. Any equation shown with the question (use LaTeX notation)
4. For multiple-choice: extract all options and identify the correct answer
5. For open-ended: provide a model answer and key terms
6. Assign 1-3 topic tags that describe what concept/skill the question tests

Tag guidelines:
- Use lowercase, specific academic terms (e.g. "normal distribution", "derivatives", "linear equations")
- Be granular: prefer "confidence intervals" over "statistics"
- Common tags: probability, hypothesis testing, normal distribution, standard deviation, \
derivatives, integrals, linear equations, quadratics, functions, percentages, \
geometry, supply and demand, elasticity, regression, correlation, matrices, \
calculus, algebra, sampling, confidence intervals, combinatorics

Return ONLY valid JSON — no markdown fences, no explanation. Format:
{
  "title": "Paper title if visible, otherwise descriptive name",
  "description": "Brief description of the paper",
  "questions": [
    {
      "id": "q1",
      "number": 1,
      "type": "multiple-choice",
      "text": "Question text here",
      "equation": "LaTeX equation or null",
      "options": [{"id": "a", "text": "Option A"}, ...],
      "correctAnswerId": "a",
      "tags": ["topic1", "topic2"]
    },
    {
      "id": "q2",
      "number": 2,
      "type": "open-ended",
      "text": "Question text here",
      "equation": null,
      "modelAnswer": "Full model answer",
      "keyTerms": ["key term 1", "key term 2"],
      "tags": ["topic1", "topic2"]
    }
  ]
}

IMPORTANT:
- Extract EVERY question, including sub-questions (label as q1a, q1b, etc.)
- If you cannot determine the correct answer for multiple-choice, make your best educated guess
- For open-ended questions, write a thorough model answer
- Preserve mathematical notation using LaTeX (\\frac, \\sqrt, ^, _, etc.)
- If the paper has multiple sections, extract from ALL sections
"""


def _extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF file."""
    import pdfplumber

    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _call_llm_extract(raw_text: str, filename: str) -> dict | None:
    """Use LLM to extract questions from raw paper text."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)

            user_prompt = (
                f"This is a past exam paper (filename: {filename}).\n\n"
                f"Raw text:\n{raw_text[:60000]}\n\n"
                "Extract all questions into the JSON format specified."
            )

            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=user_prompt,
                config={
                    "system_instruction": EXTRACT_SYSTEM_PROMPT,
                    "max_output_tokens": 32768,
                    "temperature": 0.2,
                    "thinking_config": {"thinking_budget": 8192},
                },
            )

            text = ""
            if response.candidates and response.candidates[0].content:
                for part in (response.candidates[0].content.parts or []):
                    if hasattr(part, "text") and part.text:
                        text += part.text

            if text.strip():
                return _parse_json_object(text.strip())
        except Exception:
            traceback.print_exc()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Filename: {filename}\n\nRaw text:\n{raw_text[:40000]}"},
                ],
                max_tokens=8192,
                temperature=0.2,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return _parse_json_object(text.strip())
        except Exception:
            traceback.print_exc()

    return None


def _parse_json_object(text: str) -> dict | None:
    """Parse a JSON object from LLM output, handling markdown fences and truncation."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    if start != -1:
        candidate = text[start:]
        repaired = _repair_truncated_json(candidate)
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                print(f"  [scanner] Repaired truncated JSON ({len(candidate)} → {len(repaired)} chars)")
                return result
        except json.JSONDecodeError:
            pass

    return None


def _repair_truncated_json(text: str) -> str:
    """Close open brackets/braces in truncated JSON."""
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()
    if in_string:
        text += '"'
    text = text.rstrip().rstrip(',')
    while stack:
        text += stack.pop()
    return text


def scan_paper(pdf_path: str | Path, course: str) -> Path | None:
    """Scan a single past paper PDF and save as tagged JSON.

    Returns the path to the generated JSON file, or None on failure.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"  [scanner] File not found: {pdf_path}")
        return None

    output_dir = SCANNED_OUTPUT_DIR / course
    output_dir.mkdir(parents=True, exist_ok=True)

    json_name = pdf_path.stem.replace(" ", "_") + ".json"
    output_path = output_dir / json_name

    if output_path.exists():
        print(f"  [scanner] Already scanned: {json_name}")
        return output_path

    print(f"  [scanner] Extracting text from: {pdf_path.name}")
    raw_text = _extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print(f"  [scanner] No text extracted from {pdf_path.name}")
        return None

    print(f"  [scanner] Sending to LLM for question extraction ({len(raw_text)} chars)...")
    result = _call_llm_extract(raw_text, pdf_path.name)
    if not result:
        print(f"  [scanner] LLM returned unparseable result for {pdf_path.name}")
        return None
    if "questions" not in result:
        print(f"  [scanner] LLM result missing 'questions' key for {pdf_path.name}. Keys: {list(result.keys())}")
        return None

    paper_id = f"{course.lower().replace(' ', '_')}_{pdf_path.stem.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    result["id"] = paper_id
    result["course"] = course
    result["source_file"] = pdf_path.name

    for i, q in enumerate(result["questions"]):
        if "id" not in q:
            q["id"] = f"q{i + 1}"
        if "number" not in q:
            q["number"] = i + 1
        if "tags" not in q:
            q["tags"] = []

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    q_count = len(result["questions"])
    tagged = sum(1 for q in result["questions"] if q.get("tags"))
    print(f"  [scanner] Saved {q_count} questions ({tagged} tagged) → {output_path.name}")
    return output_path


def scan_all():
    """Scan all past paper PDFs in the past_papers/ directory structure.

    Expected structure:
        past_papers/
            QM1/
                exam_2023.pdf
                exam_2024.pdf
            Economics/
                midterm_2024.pdf
    """
    if not PAST_PAPERS_DIR.exists():
        PAST_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[scanner] Created {PAST_PAPERS_DIR}/ — drop your past paper PDFs here.")
        print("[scanner] Structure: past_papers/{CourseName}/exam.pdf")
        return []

    results = []
    for course_dir in sorted(PAST_PAPERS_DIR.iterdir()):
        if not course_dir.is_dir():
            continue
        course = course_dir.name
        print(f"\n[scanner] Scanning course: {course}")

        for pdf_file in sorted(course_dir.iterdir()):
            if pdf_file.suffix.lower() != ".pdf":
                continue
            output = scan_paper(pdf_file, course)
            if output:
                results.append(output)

    print(f"\n[scanner] Done. {len(results)} papers processed.")
    return results


def load_scanned_into_db():
    """Load all scanned paper JSONs into the Paper database table."""
    from database import SessionLocal, Paper

    if not SCANNED_OUTPUT_DIR.exists():
        return

    db = SessionLocal()
    try:
        loaded = 0
        for course_dir in SCANNED_OUTPUT_DIR.iterdir():
            if not course_dir.is_dir():
                continue
            for json_file in course_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict) or "questions" not in data:
                        continue

                    paper_id = data.get("id", json_file.stem)
                    existing = db.query(Paper).filter(Paper.paper_id == paper_id).first()

                    if existing:
                        existing.title = data.get("title", "")
                        existing.description = data.get("description", "")
                        existing.course = data.get("course", course_dir.name)
                        existing.questions_json = json.dumps(data["questions"])
                        existing.question_count = len(data["questions"])
                    else:
                        paper = Paper(
                            paper_id=paper_id,
                            title=data.get("title", ""),
                            description=data.get("description", ""),
                            course=data.get("course", course_dir.name),
                            questions_json=json.dumps(data["questions"]),
                            question_count=len(data["questions"]),
                        )
                        db.add(paper)
                    loaded += 1
                except Exception:
                    traceback.print_exc()

        db.commit()
        print(f"[scanner] Loaded {loaded} scanned papers into DB")
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan past paper PDFs into tagged JSON")
    parser.add_argument("--course", help="Scan only this course directory")
    parser.add_argument("--file", help="Scan a single PDF file (requires --course)")
    parser.add_argument("--load-db", action="store_true", help="Load scanned JSONs into database")
    args = parser.parse_args()

    if args.load_db:
        load_scanned_into_db()
    elif args.file:
        if not args.course:
            print("Error: --file requires --course")
        else:
            scan_paper(args.file, args.course)
            load_scanned_into_db()
    elif args.course:
        course_dir = PAST_PAPERS_DIR / args.course
        if course_dir.is_dir():
            for pdf in sorted(course_dir.glob("*.pdf")):
                scan_paper(pdf, args.course)
            load_scanned_into_db()
        else:
            print(f"Directory not found: {course_dir}")
    else:
        scan_all()
        load_scanned_into_db()
