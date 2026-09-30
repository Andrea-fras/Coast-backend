"""Accept common citation groupings and retain only retrieved source IDs."""
import re

# [[S6], [S7]], [[S6, S7]], [[S6]], and [S6] / [S6, S7].
_ID = r'S\d+'
_ITEM = rf'\[\s*{_ID}\s*\]'
_GROUP = re.compile(
    rf'\[\s*{_ITEM}(?:\s*[,;]\s*{_ITEM})*\s*\]'
    rf'|\[\[\s*{_ID}(?:\s*[,;]\s*{_ID})*\s*\]\]'
    rf'|\[\s*{_ID}(?:\s*[,;]\s*{_ID})*\s*\]'
)
_IDS = re.compile(_ID)
# Keep examples in Markdown code and LaTeX literal, matching the UI's AST rules.
_LITERAL = re.compile(
    r'(`+)[\s\S]*?\1|(?m:^[ ]{0,3}(~{3,})[^\n]*\n[\s\S]*?^[ ]{0,3}\2[ \t]*(?:\n|$))'
    r'|(\${1,2})[\s\S]*?\3'
)


def _rewrite(text, render):
    pieces, start = [], 0
    for literal in _LITERAL.finditer(text):
        pieces.extend((_GROUP.sub(render, text[start:literal.start()]), literal.group(0)))
        start = literal.end()
    pieces.append(_GROUP.sub(render, text[start:]))
    return ''.join(pieces)


def normalize_citations(text, allowed):
    """Canonicalize groups before filtering/persisting the citation catalogue."""
    used = set()

    def render(match):
        ids = dict.fromkeys(_IDS.findall(match.group(0)))
        valid = [sid for sid in ids if sid in allowed]
        used.update(valid)
        return ' '.join(f'[[{sid}]]' for sid in valid)

    return _rewrite(text, render), used


def strip_citations(text):
    """Old source IDs must not be reused as evidence for the next question."""
    return _rewrite(text, lambda _: '')
