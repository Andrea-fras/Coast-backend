"""The container that reads and indexes course files (see coast_content_oma/remote.py).

    python -m modal deploy coast_modal.py     # Render's build runs this with every push

Its settings and keys are taken from the environment at deploy time (Render's, or .env locally),
so the container always runs with what the server runs with.
"""
import os

import modal

from coast_content_oma.remote import APP

try:  # locally the keys live in .env; on Render they are already in the environment
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

SETTINGS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
            "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET",
            "OMA_READER", "OMA_OPENAI_MODEL", "OMA_OPENAI_VISION_MODEL", "OMA_GEMINI_MODEL", "OMA_GEMINI_VISION_MODEL",
            "OMA_DESCRIBE_IMAGES", "OMA_INLINE_FIGURES", "OMA_VISION_BATCH_SIZE", "OMA_OCR_LANGUAGE",
            "COAST_AI_MAX_WAITERS", "COAST_CREDIT_COOLDOWN_SEC")
env = {k: os.environ[k] for k in SETTINGS if os.environ.get(k)}
# Files sit at the server's paths (each call says where), fetched from and published to R2; one
# file per container, so its AI calls may use the whole provider allowance of the container.
env.update(FILE_STORE="on", FILE_STORE_THREADS="32", COAST_FIGURE_THREADS="2", COAST_OPENAI_CONCURRENCY="16",
           COAST_OPENAI_BACKGROUND_CONCURRENCY="12", COAST_GEMINI_CONCURRENCY="12",
           COAST_GEMINI_BACKGROUND_CONCURRENCY="9", COAST_AI_USAGE="off")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tesseract-ocr", "tesseract-ocr-eng")  # text for scanned pages
    .env({"TESSDATA_PREFIX": "/usr/share/tesseract-ocr/5/tessdata"})
    .pip_install("PyMuPDF>=1.24.0", "Pillow>=10.0.0", "pdfplumber>=0.11.0", "pypdf", "python-pptx>=0.6.23",
                 "openai>=1.30.0", "google-genai>=1.0.0", "boto3>=1.34", "numpy>=1.26.0", "python-dotenv>=1.0.0",
                 "requests", "psutil>=5.9")
    .add_local_python_source("coast_content_oma", "file_store", "backups", "provider_capacity", "ai_usage", "memory_budget")
)
app = modal.App(APP, image=image, secrets=[modal.Secret.from_dict(env)])


@app.cls(cpu=2.0, memory=1024, timeout=1800, max_containers=100,  # two CPUs save figures twice as fast
         scaledown_window=60, enable_memory_snapshot=True)
class Indexer:
    """One pool of containers for both jobs, so the container that just read a file can index it
    next (an idle one stays up a minute: idle time is billed like work), and a new one starts from a
    snapshot taken with the PDF and AI libraries already loaded (a burst of 20 cold starts took
    15-20 s without)."""

    @modal.enter(snap=True)
    def load(self):
        import boto3, fitz, google.genai, numpy, openai, pdfplumber, PIL.Image  # noqa: F401
        import coast_content_oma.extraction, coast_content_oma.ingestion, coast_content_oma.remote  # noqa: F401

    @modal.method()
    def read_upload(self, path: str, context: dict) -> dict:
        from coast_content_oma.remote import read_upload_work
        return read_upload_work(path, context)

    @modal.method()
    def index_pages(self, job: dict):
        from coast_content_oma.remote import index_work
        yield from index_work(job)
