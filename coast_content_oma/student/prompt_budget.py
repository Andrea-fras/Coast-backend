"""Rank structured evidence before fitting the prompt budget."""
import re

RULES = ('STUDENT HISTORY: Use only recorded evidence; never invent past events. '
         'A single mistake is not a persistent weakness. Assisted success differs from independent mastery. '
         'Reuse relevant examples naturally; omit unrelated memories.')

def brief(value, limit=180):
    value = re.sub(r'\s+', ' ', str(value or '')).strip()
    return value if len(value) <= limit else value[:limit-1].rstrip() + '…'

def pack(entries, max_chars):
    if max_chars <= 0:
        return ''
    # Rules cannot disappear behind a long accomplishment ledger.
    out = [RULES[:max_chars]]
    remaining = max_chars - len(out[0])
    seen = set()
    for entry in entries:
        entry = brief(entry, 250)
        if not entry or entry in seen or len(entry) + 1 > remaining:
            continue
        out.append(entry)
        seen.add(entry)
        remaining -= len(entry) + 1
    return '\n'.join(out)

def _state(m):
    state = (m.get('misconception_state') or m.get('last_eval_state') or 'UNVERIFIED').upper()
    return {'MISCONCEPTION': 'ACTIVE_MISCONCEPTION', 'STRUGGLING': 'UNDER_OBSERVATION',
            'MASTERED': 'RESOLVED'}.get(state, state)


_TRAIT_LABEL = {'general_strength': 'Cross-course strength', 'general_weakness': 'Cross-course difficulty'}


def _trait_line(trait, limit):
    return f"{_TRAIT_LABEL.get(trait.get('type'), 'Observed preference')}: {brief(trait.get('text'), limit)}"


def _style_line(profile):
    """Learning-style traits as an instruction, with the student's own words."""
    styles = [t for t in profile.get('identity_traits') or [] if t.get('type') == 'learning_style']
    if not styles:
        return None
    parts = []
    for t in styles[:2]:
        part = brief(t.get('text'), 90)
        if t.get('quote'):
            part += f" (they said: \"{brief(t['quote'], 110)}\")"
        parts.append(part)
    return 'HOW TO TEACH THIS STUDENT: ' + '; '.join(parts) + ' — apply it from your first message, without announcing it.'


def _prior_lines(profile, limit=2):
    return [f"Builds on {brief(p.get('course'), 45)} ({p.get('when') or 'earlier'}): they worked on "
            f"{brief(p.get('concept'), 50)} ({p.get('state')}) — relates to {brief(p.get('relates_to'), 50)} here."
            for p in (profile.get('prior_learning') or [])[:limit]]


def _relevant_mistakes(profile, focus_ids):
    """Unresolved wrong answers on the concepts being taught (else the current section)."""
    current = (profile.get('progress_ledger') or {}).get('current_section')
    mistakes = profile.get('section_mistakes') or []
    hits = [m for m in mistakes if focus_ids.intersection(m.get('concept_ids') or [])]
    if not hits and current is not None:
        hits = [m for m in mistakes if m.get('section_index') == current]
    return hits[-2:]


def _strong(m):
    return (_state(m) == 'RESOLVED' and m.get('effective_score', m.get('score', 0)) >= .85
            and m.get('confidence', 0) >= .7 and m.get('successes', 0) >= 3)


def course_profile(profile, max_chars=1200):
    entries = [e for e in [_style_line(profile)] if e] + _prior_lines(profile)
    focused = profile.get('focused_mastery') or []
    focus_ids = {m.get('concept_id') for m in focused if m.get('concept_id')}
    focus_ids.update(profile.get('requested_concept_ids') or [])
    query_terms = set(re.findall(r'\w+', (profile.get('current_query') or '').lower()))
    def priority(m):
        urgent = {'ACTIVE_MISCONCEPTION': 0, 'UNDER_OBSERVATION': 1}.get(_state(m), 2)
        match = len(query_terms.intersection(re.findall(r'\w+', (m.get('name') or '').lower())))
        return (urgent, -match, m.get('effective_score', m.get('score', 0)), m.get('name') or '')
    focused = sorted(focused, key=priority)
    requested = max(profile.get('requested_concept_count', 0), len(focused))
    if requested:
        entries.append(f'Relevant evidence: {len(focused)}/{requested} concepts recorded; '
                       f'{sum(_strong(m) for m in focused)} strong/resolved; '
                       f'{requested-len(focused)} unassessed. Scores are evidence, not proof of current mastery.')
    if focused and _state(focused[0]) == 'ACTIVE_MISCONCEPTION':
        entries.append(f"Teaching stance: REPAIR {brief(focused[0].get('name'), 45)} first. "
                       'Use a minimal contrast and one short probe before the full algorithm or next topic.')
    elif requested and len(focused) == requested and all(_strong(m) for m in focused):
        entries.append('Teaching stance: REVIEW. Open with 2–3 recap sentences TOTAL, then one challenging '
                       'transfer question. No beginner lecture or worked solution unless the answer reveals a gap.')
    cards = []
    for m in focused[:3]:
        cards.append(f"Current concept: {brief(m.get('name'), 55)}; {_state(m)}; "
                     f"mastery {m.get('effective_score', m.get('score', 0)):.2f}; confidence {m.get('confidence', 0):.2f}; "
                     f"{m.get('successes', 0)} successful responses, {m.get('struggles', 0)} struggles.")
    if cards:
        entries.append(cards[0])
    active = profile.get('active_context') or {}
    if active.get('last_unresolved'):
        entries.append('Unresolved: ' + brief(active['last_unresolved'].get('text'), 160))
    elif focused and focused[0].get('misconception_type') and _state(focused[0]) == 'ACTIVE_MISCONCEPTION':
        entries.append('Unresolved: ' + brief(focused[0]['misconception_type'], 160))
    for m in _relevant_mistakes(profile, focus_ids):
        section = m.get('section_index')
        where = f"Section {section + 1}" if isinstance(section, int) else 'earlier'
        entries.append(f"Unresolved mistake ({where}, {m.get('when') or 'date unknown'}): "
                       f"\"{brief(m.get('user_message'), 150)}\" — check it is fixed before moving on.")
    traits = sorted((t for t in profile.get('identity_traits') or [] if t.get('type') != 'learning_style'),
                    key=lambda t: t.get('confidence', 0), reverse=True)
    for trait in traits[:1]:
        entries.append(_trait_line(trait, 145))
    golden = profile.get('golden_moments') or []
    if focus_ids:
        # No fallback to an unrelated analogy merely because it is available.
        golden = [g for g in golden if focus_ids.intersection(g.get('concept_ids') or [])]
    recall = [f"From {brief(g.get('folder'), 45)}: {brief(g.get('text'), 165)}"
              for g in (profile.get('cross_course_memories') or [])[:2]]
    explicit_recall = bool(query_terms.intersection({'remember', 'recall', 'previous', 'semester', 'yesterday'}))
    if explicit_recall:
        entries.extend(recall[:1])
    for g in golden[:1]:
        entries.append('Worked before: ' + brief(g.get('text'), 175))
    if not explicit_recall:
        entries.extend(recall[:1])
    entries.extend(cards[1:])
    ledger = profile.get('progress_ledger') or {}
    completed = ledger.get('completed_sections') or []
    if completed:
        entries.append(f"Completed {len(completed)} sections. Most recent: {brief(completed[-1].get('title'), 95)}.")
    if ledger.get('current_section_title'):
        entries.append('Current section: ' + brief(ledger['current_section_title'], 95))
    for s in (profile.get('struggling_topics') or [])[:2]:
        entries.append(f"Repeated difficulty: {brief(s.get('name'), 65)}; {s.get('mistakes', 0)} wrong / {s.get('successes', 0)} right.")
    for trait in traits[1:2]:
        entries.append(_trait_line(trait, 110))
    for item in (profile.get('due_for_review') or [])[:2]:
        entries.append('Consider a refresher: ' + brief(item.get('name'), 90))
    for c in (profile.get('other_courses') or [])[:2]:
        entries.append(f"Other course: {brief(c.get('folder'), 55)}; {c.get('sections_completed', 0)} sections completed.")
    for p in (profile.get('course_patterns') or [])[:2]:
        entries.append('Course observation: ' + brief(p.get('text'), 120))
    return pack(entries, max_chars)

def global_profile(profile, max_chars=1600):
    entries = [e for e in [_style_line(profile)] if e]
    for g in (profile.get('cross_course_memories') or [])[:3]:
        entries.append(f"Recall from {brief(g.get('folder'), 45)}: {brief(g.get('text'), 160)}")
    for t in [t for t in profile.get('identity_traits') or [] if t.get('type') != 'learning_style'][:2]:
        entries.append(_trait_line(t, 100))
    # One concise entry per course before adding details from any one course.
    courses = profile.get('courses') or []
    for c in courses:
        acc = (c.get('accomplishments') or {}).get('narrative_lines') or []
        detail = acc[-1] if acc else c.get('current_focus') or 'No detailed evidence recorded'
        entries.append(f"Course {brief(c.get('folder'), 45)}: {brief(detail, 130)}")
    for c in courses:
        for p in (c.get('patterns') or [])[:1]:
            entries.append(f"In {brief(c.get('folder'), 45)}: {brief(p.get('text'), 130)}")
    return pack(entries, max_chars)
