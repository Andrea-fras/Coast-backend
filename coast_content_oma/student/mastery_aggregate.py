"""Virtual mastery aggregation across alias concept IDs.

When the alias ledger maps con_47 → con_12, both may have separate mastery
rows. Read sites call aggregate_mastery_rows() once, consistently, instead
of each rolling their own fold.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .mastery_tier import sync_mastery_tier
from .stores.concept_mastery import CONFIDENCE_CAP_N, effective_mastery


def evidence_count(ss: dict) -> int:
    return int(ss.get("successes", 0) or 0) + int(ss.get("struggles", 0) or 0) + int(
        ss.get("neutral_touches", 0) or 0
    )


def eval_severity(ss: dict) -> int:
    """Higher = needs Pedro attention. Pessimistic-wins for conflicts."""
    state = (ss.get("misconception_state") or ss.get("last_eval_state") or "").lower()
    if ss.get("last_misconception") or state in ("misconception", "active_misconception"):
        return 4
    if state in ("struggling", "under_observation"):
        return 3
    if state == "resolved":
        return 2
    if state == "mastered":
        return 1
    return 0


_SEVERITY_TO_STATE = {
    4: "misconception",
    3: "struggling",
    2: "resolved",
    1: "mastered",
    0: None,
}


def _min_ts(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a and b:
        return min(a, b)
    return a or b


def _max_ts(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a and b:
        return max(a, b)
    return a or b


def aggregate_mastery_rows(
    rows: list[dict],
    *,
    canonical_id: str,
    canonical_name: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """Combine multiple mastery store_specific dicts into one virtual row.

    Rules (mirror merge_concepts write-time fold, plus state precedence):
      - mastery_score: evidence-weighted average of raw scores
      - effective score: evidence-weighted average of per-row effective_mastery
      - counters: summed
      - timestamps: earliest first_seen, latest last_seen / strengthened / struggle
      - last_misconception: OR across rows
      - last_eval_state: pessimistic-wins (misconception > struggling > resolved > mastered)
    """
    if not rows:
        return None

    total_weight = 0
    weighted_raw = 0.0
    weighted_eff = 0.0
    agg: dict = {
        "concept_id": canonical_id,
        "concept_name": canonical_name or rows[0].get("concept_name") or canonical_id,
        "successes": 0,
        "struggles": 0,
        "neutral_touches": 0,
        "related_lesson_ids": [],
    }
    best_severity = 0
    best_eval_section: Optional[int] = None

    for ss in rows:
        n = evidence_count(ss)
        w = max(1, n) if n > 0 else 1
        raw = float(ss.get("mastery_score", 0.0) or 0.0)
        eff = effective_mastery(ss, now)
        weighted_raw += raw * w
        weighted_eff += eff * w
        total_weight += w

        for key in ("successes", "struggles", "neutral_touches"):
            agg[key] = agg.get(key, 0) + int(ss.get(key, 0) or 0)

        agg["first_seen"] = _min_ts(agg.get("first_seen"), ss.get("first_seen"))
        agg["last_seen"] = _max_ts(agg.get("last_seen"), ss.get("last_seen"))
        agg["last_strengthened"] = _max_ts(agg.get("last_strengthened"), ss.get("last_strengthened"))
        agg["last_struggle"] = _max_ts(agg.get("last_struggle"), ss.get("last_struggle"))

        if ss.get("last_misconception"):
            agg["last_misconception"] = True

        sev = eval_severity(ss)
        if sev > best_severity:
            best_severity = sev
            agg.pop("misconception_type", None)
            if ss.get("misconception_type"):
                agg["misconception_type"] = ss["misconception_type"]
            best_eval_section = (int(ss["last_eval_section"])
                                 if ss.get("last_eval_section") is not None else None)

        if not agg.get("concept_name") and ss.get("concept_name"):
            agg["concept_name"] = ss["concept_name"]

        lessons = list(dict.fromkeys(
            (agg.get("related_lesson_ids") or []) + (ss.get("related_lesson_ids") or [])
        ))
        agg["related_lesson_ids"] = lessons

    if total_weight <= 0:
        total_weight = len(rows)

    agg["mastery_score"] = max(0.0, min(1.0, weighted_raw / total_weight))
    agg["effective_score"] = max(0.0, min(1.0, weighted_eff / total_weight))
    agg["confidence"] = min(1.0, evidence_count(agg) / CONFIDENCE_CAP_N)

    winning_state = _SEVERITY_TO_STATE.get(best_severity)
    if winning_state:
        agg["last_eval_state"] = winning_state
        agg['misconception_state'] = {4: 'ACTIVE_MISCONCEPTION', 3: 'UNDER_OBSERVATION',
                                      2: 'RESOLVED', 1: 'RESOLVED'}[best_severity]
        if best_eval_section is not None:
            agg["last_eval_section"] = best_eval_section

    sync_mastery_tier(agg)
    return agg
