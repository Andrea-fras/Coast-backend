"""python -m coast_content_oma.read_upload <file>

Reads an uploaded PDF or PowerPoint in its own process (see server._read_upload): saves the
normalized page copy next to the file and prints the pages' text as JSON on stdout.
"""
import json
import sys

from .extraction import extract_and_cache

if __name__ == "__main__":
    json.dump(extract_and_cache(sys.argv[1]), sys.stdout)
