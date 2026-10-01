"""python -m coast_content_oma.read_upload <file>

Reads an uploaded PDF or PowerPoint in its own process (see server._read_upload): saves the
normalized page copy next to the file and prints the pages' text as JSON on stdout.

Only the JSON may reach stdout. MuPDF writes its messages there from C ("MuPDF error:
cannot create appearance stream for Screen annotations" for a slide with an embedded
video), so while the file is read, file descriptor 1 points at stderr and the JSON goes
out on a copy of the real stdout.
"""
import json
import os
import sys


def main(path: str, extract=None) -> None:
    results = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)           # C libraries writing to stdout now write to stderr
    sys.stdout = sys.stderr  # and so does Python's print
    if extract is None:
        from .extraction import extract_and_cache as extract
    json.dump(extract(path), results)
    results.flush()


if __name__ == "__main__":
    main(sys.argv[1])
