# OCR Past Paper Pipeline

Convert any past exam paper (PDF or image) into structured JSON with full support for:

- **Math formulas** → extracted as LaTeX strings
- **Diagrams & images** → detected and described
- **Multiple-choice & open-ended questions** → classified automatically
- **Model answers & key terms** → generated when possible

## How It Works

The pipeline uses **GPT-4o's vision capabilities** to analyze scanned exam pages. Unlike traditional OCR, this approach understands document structure, mathematical notation, and diagram content natively.

```
Input (PDF/Image) → Page Images → GPT-4o Vision → Structured JSON → Validated Output
```

## Setup

### 1. Install Dependencies

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

### 2. Install Poppler (required for PDF processing)

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install poppler-utils

# Windows – download from:
# https://github.com/oschwartz10612/poppler-windows
```

### 3. Set Your API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Scan a Single Paper

```bash
# Basic usage – outputs to <filename>_output/<filename>.json
python main.py scan exam_paper.pdf

# Specify output path
python main.py scan exam_paper.pdf -o output/exam.json

# Save page images alongside JSON
python main.py scan exam_paper.pdf --save-pages

# Add extra context for better extraction
python main.py scan exam_paper.pdf --instructions "This is an A-Level Physics paper"
```

### Batch Process Multiple Papers

```bash
python main.py batch paper1.pdf paper2.pdf paper3.png -o output/
```

### View Previously Extracted JSON

```bash
python main.py show output/exam.json
```

### Use as a Python Module

```python
from pipeline import run_pipeline

result = run_pipeline(
    "exam_paper.pdf",
    output_path="output/exam.json",
    extra_instructions="This is a GCSE Maths paper",
    save_page_images=True,
)

print(f"Extracted {len(result['questions'])} questions")
```

## Output Format

```json
{
  "id": "econ_101",
  "title": "Economics 101",
  "description": "Introduction to Microeconomics",
  "questions": [
    {
      "id": "q1",
      "number": 1,
      "type": "multiple-choice",
      "text": "What happens to demand when...",
      "equation": null,
      "images": null,
      "options": [
        { "id": "a", "text": "Demand decreases" },
        { "id": "b", "text": "Demand increases" }
      ],
      "correctAnswerId": "b"
    },
    {
      "id": "q2",
      "number": 2,
      "type": "open-ended",
      "text": "Explain the concept of...",
      "equation": "E = mc^2",
      "images": ["q2_diagram.png"],
      "modelAnswer": "The concept refers to...",
      "keyTerms": ["key term 1", "key term 2"]
    }
  ]
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Slug identifier for the paper |
| `title` | string | Paper title |
| `description` | string \| null | Short description |
| `questions[].type` | string | `"multiple-choice"` or `"open-ended"` |
| `questions[].equation` | string \| null | LaTeX formula if present |
| `questions[].images` | string[] \| null | Diagram filenames if present |
| `questions[].options` | object[] | MCQ answer options |
| `questions[].correctAnswerId` | string \| null | Correct option letter |
| `questions[].modelAnswer` | string \| null | Ideal answer (open-ended) |
| `questions[].keyTerms` | string[] \| null | Expected terms (open-ended) |

## Displaying on Your Platform

### Math Formulas (LaTeX)

The `equation` field contains standard LaTeX. Render with:
- **React**: [react-katex](https://www.npmjs.com/package/react-katex) or [react-mathjax](https://www.npmjs.com/package/react-mathjax)
- **Web**: [KaTeX](https://katex.org/) or [MathJax](https://www.mathjax.org/)

```jsx
import { InlineMath, BlockMath } from 'react-katex';
<BlockMath math={question.equation} />
```

### Diagrams

When diagrams are detected, the JSON includes:
- `images`: filenames for the diagram images
- `imageDescriptions`: textual descriptions of what each diagram shows

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o` | Model to use for vision OCR |
