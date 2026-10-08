#!/usr/bin/env python3
"""10-lesson adaptation eval: real Pedro, simulated student, real Student OMA.

A lesson = one whole course from its own sources, every section passed through
Pedro's comprehension gate. After each lesson we record what memory holds, and
at the start of the next lesson an LLM judge scores whether Pedro adapted.
Recall questions in general chat check exact memory at set points.

Runs against a COPY of the local databases in eval_runs/<name>/ — never the real ones.
Resumable: re-run the same command to continue after an interruption.

  python3 scripts/adaptation_eval/run.py --name first_run
  python3 scripts/adaptation_eval/run.py --name smoke --lessons 1 --max-sections 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _prepare(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in ((ROOT / "coast.db", run_dir / "coast.db"), (ROOT / "oma_data" / "oma.db", run_dir / "oma.db")):
        if not dst.exists():
            with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
                s.backup(d)
    os.environ.update({
        "DATABASE_PATH": str(run_dir / "coast.db"),
        "OMA_DB_PATH": str(run_dir / "oma.db"),
        "OMA_IMAGE_DIR": str(run_dir / "images"),
        "RAG_PROVIDER": "oma",
        "STUDENT_OMA_ENABLED": "true",
        "OMA_DESCRIBE_IMAGES": "false",
    })
    os.environ.pop("RENDER", None)


ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="run name (eval_runs/<name>)")
ap.add_argument("--lessons", type=int, default=10, help="how many lessons to run")
ap.add_argument("--max-sections", type=int, default=None, help="cap sections per lesson (smoke tests)")
ap.add_argument("--max-turns", type=int, default=12, help="student turns per section before forcing the gate")
ap.add_argument("--pedro", choices=["gemini", "anthropic"], default=None,
                help="which provider plays Pedro (default: PEDRO_PROVIDER from .env)")
ARGS = ap.parse_args()
if ARGS.pedro:
    os.environ["PEDRO_PROVIDER"] = ARGS.pedro  # before tutor is imported
RUN_DIR = ROOT / "eval_runs" / ARGS.name
_prepare(RUN_DIR)

os.chdir(ROOT)
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import database  # noqa: E402

assert Path(database.DB_PATH).resolve() == (RUN_DIR / "coast.db").resolve(), database.DB_PATH
database.init_db()
import tutor  # noqa: E402
import lesson  # noqa: E402
import learning_jobs  # noqa: E402
import oma_provider  # noqa: E402
from database import SessionLocal, User, LearningJob  # noqa: E402
from coast_content_oma.student.stores import course_namespace, identity_namespace  # noqa: E402
from coast_content_oma.student.stores.academic_identity import is_displayable_trait  # noqa: E402
from scripts.persona_audit.digital_twin.student_agent import StudentAgent  # noqa: E402

from scenario import BEATS, LESSONS, PROBES, STUDENT  # noqa: E402
from courses import copy_course  # noqa: E402
import judge  # noqa: E402

assert Path(oma_provider.OMA_DB_PATH).resolve() == (RUN_DIR / "oma.db").resolve()
STATE_PATH = RUN_DIR / "state.json"
LOG = open(RUN_DIR / "progress.log", "a", buffering=1)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


# ── Capture what Pedro is given ──────────────────────────────────────

_last_prompt: dict = {}
_orig_build = tutor.build_system_prompt


def _capturing_build(*args, **kwargs):
    out = _orig_build(*args, **kwargs)
    _last_prompt.clear()
    _last_prompt["profile"] = kwargs.get("student_profile_block") or ""
    blocks = re.findall(r"--- STUDENT OMA[\s\S]*?--- END STUDENT OMA ---", kwargs.get("notebook_content") or "")
    _last_prompt["section_blocks"] = blocks
    return out


tutor.build_system_prompt = _capturing_build


# ── Simulated student with planted moments ──────────────────────────

class ScenarioStudent(StudentAgent):
    def __init__(self, state):
        super().__init__("visual_learner")
        self.persona = dict(STUDENT)
        self.lesson_n = None
        self.state = state

    def pending_beats(self):
        done = self.state["beats"]
        return [b for b in BEATS if b["id"] not in done and b["lesson"] == self.lesson_n
                and b["section"] in (None, self.current_section)]

    def _simulator_prompt(self, pedro_message, recent_turns):
        base = super()._simulator_prompt(pedro_message, recent_turns)
        beats = self.pending_beats()
        if not beats:
            return base
        extra = "\n".join(f"- [{b['id']}] {b['instruction']}" for b in beats)
        return base + (
            "\n\nSCRIPTED MOMENTS for this part of the course — do each ONCE, naturally, when it fits "
            "(not necessarily this turn). Stay honest otherwise:\n" + extra +
            '\nWhen your Part-1 reply performs one, add "scripted_moment_done": "<its id>" to the JSON.'
        )


# ── Helpers ─────────────────────────────────────────────────────────

class ProviderDown(RuntimeError):
    """A model call failed — the turn is not real tutoring and must not be scored."""


_FALLBACKS = ("I'm having a brief technical issue", "I'd love to help with that! Could you rephrase")


def wait_for_network(max_wait=1800) -> None:
    import socket
    import ssl
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    deadline = time.time() + max_wait
    while True:
        try:  # full TLS handshake: a captive portal accepts TCP but fails here
            for host in ("generativelanguage.googleapis.com", "api.openai.com"):
                with socket.create_connection((host, 443), timeout=5) as sock:
                    ctx.wrap_socket(sock, server_hostname=host).close()
            return
        except OSError:
            if time.time() > deadline:
                raise SystemExit("Network still down after 30 min — re-run the same command to resume.")
            log("  network down; waiting…")
            time.sleep(30)


def chat(uid, message, *, context_type="lesson", context_id=None, section_index=None, conversation_id=None):
    final = None
    for _tok, res in tutor.send_message_stream(uid, message, conversation_id, context_type,
                                               context_id=context_id, section_index=section_index):
        if res is not None:
            final = res
    oma_provider.flush_student_writes()
    if any(f in ((final or {}).get("reply") or "") for f in _FALLBACKS):
        raise ProviderDown("Pedro returned a fallback reply")
    return final or {}


def discard_conversation(uid, conversation_id) -> None:
    """Remove an aborted attempt (chat rows + the memory records pointing at them)."""
    from database import ChatMessage
    from coast_content_oma.stores.db import connect_db
    with SessionLocal() as db:
        ids = [m.id for m in db.query(ChatMessage).filter_by(user_id=uid, conversation_id=conversation_id)]
        db.query(ChatMessage).filter_by(user_id=uid, conversation_id=conversation_id).delete()
        db.commit()
    if ids:
        with connect_db(oma_provider.OMA_DB_PATH) as conn:
            marks = ",".join("?" * len(ids))
            doomed = [r[0] for r in conn.execute(
                f"SELECT DISTINCT e.id FROM episode_items e, json_each(e.store_specific, '$.chat_message_ids') j "
                f"WHERE e.namespace LIKE ? AND j.value IN ({marks})", (f"u{uid}__%", *ids))]
            for eid in doomed:
                conn.execute("DELETE FROM episode_items WHERE id = ?", (eid,))
                from coast_content_oma.stores.db import fts_rowid
                conn.execute("DELETE FROM episode_items_fts WHERE rowid = ?", (fts_rowid(eid),))


def settle_jobs(uid, timeout=240) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if learning_jobs.run_one(kind="completion"):
            continue
        with SessionLocal() as db:
            pending = [j for j in db.query(LearningJob).filter(LearningJob.status.in_(("queued", "running")))
                       if json.loads(j.payload_json).get("user_id") == uid and not json.loads(j.payload_json).get("kind")]
        if not pending:
            return True
        time.sleep(3)
    return False


def ledger_text(state) -> str:
    lines = []
    for n in sorted(state["lessons"], key=int):
        L = state["lessons"][n]
        if L.get("done"):
            lines.append(f"- Completed course {n} \"{L['name']}\" ({L['sections']} sections; "
                         f"{L['gates_passed']} gates passed by the student, {L['gates_forced']} forced).")
    for bid, b in state["beats"].items():
        lines.append(f"- In course {b['lesson']} \"{b['lesson_name']}\" section {b['section'] + 1}: {b['memory']} "
                     f"(the student said: \"{b['quote'][:200]}\")")
    return "\n".join(lines) or "- Nothing yet: this is a brand-new student."


def memory_snapshot(uid, state) -> dict:
    orch = oma_provider._student_orchestrator()
    snap = {"courses": {}, "identity_traits": [], "junk_traits": 0}
    for n, L in state["lessons"].items():
        if not L.get("setup"):
            continue
        ns = course_namespace(uid, L["name"])
        eps = orch.episodes.all(ns)
        kinds = {}
        for e in eps:
            k = (e.store_specific or {}).get("episode_type")
            kinds[k] = kinds.get(k, 0) + 1
        mastery = sorted(((it.store_specific or {}).get("concept_name"), round(float((it.store_specific or {}).get("mastery_score", 0)), 2),
                          (it.store_specific or {}).get("mastery_tier"), (it.store_specific or {}).get("last_eval_state"))
                         for it in orch.mastery.all(ns))
        patterns = [(p.store_specific or {}).get("pattern_type") + ": " + p.content for p in orch.patterns.all(ns)]
        snap["courses"][L["name"]] = {"episodes": kinds, "mastery": mastery, "patterns": patterns,
                                      "open_questions": [q["text"] for q in orch.active.snapshot(ns)["open_questions"]]}
    for it in orch.identity.all(identity_namespace(uid)):
        if is_displayable_trait(it.content):
            snap["identity_traits"].append(f"{(it.store_specific or {}).get('trait_type')}: {it.content}")
        else:
            snap["junk_traits"] += 1
    return snap


def planted_in_memory(uid, state, snap) -> dict:
    """Is each planted moment that actually happened represented in Student OMA?"""
    out = {}
    text_all = json.dumps(snap).lower()
    for bid, b in state["beats"].items():
        course = snap["courses"].get(b["lesson_name"], {})
        blob = json.dumps(course).lower()
        if bid == "prefers_examples":
            ok = any("example" in t.lower() for t in snap["identity_traits"]) or "prefers_examples" in blob
        elif bid == "markov_misconception":
            ok = course.get("episodes", {}).get("exercise_attempt", 0) > 0 and ("mistake" in blob or "misconception" in blob
                 or "struggling" in blob or "column" in blob) or "column" in blob
        elif bid == "steady_state_analogy":
            ok = "golden_moment" in blob
        elif bid == "exam_goal":
            ok = "exam" in text_all
        elif bid == "crossover_confusion":
            ok = "crossover" in blob or "mutation" in blob
        else:
            ok = None
        out[bid] = ok
    return out


def save(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ── Main loop ───────────────────────────────────────────────────────

def run_section(uid, agent, L, idx, sec, state) -> dict:
    for attempt in range(1, 6):
        wait_for_network()
        conv = f"eval_{uid}_{L['n']}_{idx}_a{attempt}"
        beats_before = dict(state["beats"])
        try:
            return _run_section(uid, agent, L, idx, sec, state, conv)
        except ProviderDown as exc:
            log(f"    section {idx + 1} attempt {attempt} aborted ({exc}); discarding and retrying")
            discard_conversation(uid, conv)
            state["beats"] = beats_before
            agent.state = state
            time.sleep(60)
    raise SystemExit("Providers kept failing — re-run the same command to resume.")


def _run_section(uid, agent, L, idx, sec, state, conv) -> dict:
    title = sec.get("title") or f"Section {idx + 1}"
    refs = lesson.get_section_concept_refs(uid, L["name"], idx)
    agent.start_section(idx, title, [r["concept_name"] for r in refs])
    turns = []
    res = chat(uid, f'I\'m ready to learn about "{title}". Please teach me this section.',
               context_id=L["name"], section_index=idx, conversation_id=conv)
    first_prompt = dict(_last_prompt)
    reply = res.get("reply") or ""
    turns.append({"role": "pedro", "text": oma_provider.strip_pedro_tags(reply)})
    passed = "[SECTION_COMPLETE]" in reply
    for t in range(ARGS.max_turns):
        if passed:
            break
        msg, update = agent.respond(oma_provider.strip_pedro_tags(reply), recent_turns="\n".join(
            f"{x['role']}: {x['text'][:300]}" for x in turns[-4:]))
        if not msg:
            raise ProviderDown("student simulator returned nothing")
        turns.append({"role": "student", "text": msg})
        bid = (update or {}).get("scripted_moment_done")
        beat = next((b for b in BEATS if b["id"] == bid), None)
        if beat and bid not in state["beats"]:
            state["beats"][bid] = {"lesson": L["n"], "lesson_name": L["name"], "section": idx,
                                   "memory": beat["memory"], "quote": msg}
            log(f"    planted {bid}: {msg[:90]!r}")
        res = chat(uid, msg, context_id=L["name"], section_index=idx, conversation_id=conv)
        reply = res.get("reply") or ""
        turns.append({"role": "pedro", "text": oma_provider.strip_pedro_tags(reply)})
        passed = "[SECTION_COMPLETE]" in reply
    if not lesson.can_advance_from_section(uid, L["name"], idx):
        lesson.mark_section_verified(uid, L["name"], idx)
    adv = lesson.advance_section(uid, L["name"])
    evaluated = settle_jobs(uid)
    agent.end_section()
    log(f"    section {idx + 1} '{title[:50]}': {len(turns) // 2} student turns, "
        f"gate {'PASSED' if passed else 'FORCED'}, evaluator {'ok' if evaluated else 'PENDING'}"
        f"{'' if 'error' not in adv else ' advance error: ' + str(adv['error'])}")
    return {"title": title, "turns": turns, "gate_passed": passed, "evaluated": evaluated,
            "prompt_at_opener": first_prompt}


def judge_start(L, spec, state) -> None:
    """Score Pedro's first messages in this lesson against what happened before it."""
    if L.get("start_judgement") or not L["sections_log"]:
        return
    first = L["sections_log"][0]
    excerpt = "\n".join(f"{t['role'].upper()}: {t['text'][:1200]}" for t in first["turns"][:7])
    related = ", ".join(f"course {r} \"{LESSONS[r - 1]['name']}\"" for r in (spec["related_to"] or [])) \
        or "none — this course is unrelated to earlier ones"
    wait_for_network()
    L["start_judgement"] = judge.judge_lesson_start(L["ledger_before"], spec["name"], related, excerpt)
    L["pedro_was_given"] = first["prompt_at_opener"]
    save(state)
    j = L["start_judgement"]
    log(f"  judge @ start: continuity={j.get('continuity', {}).get('score')} "
        f"preference={j.get('preference', {}).get('score')} memory={j.get('relevant_memory', {}).get('score')} "
        f"fabrication={j.get('fabrication', {}).get('found')} — {j.get('verdict', '')[:120]}")


def run_probe(uid, spec, probe, state) -> None:
    for attempt in range(1, 6):
        wait_for_network()
        conv = f"probe_{uid}_{spec['n']}_a{attempt}"
        try:
            res = chat(uid, probe["question"], context_type="global", conversation_id=conv)
            break
        except ProviderDown:
            discard_conversation(uid, conv)
            time.sleep(60)
    else:
        raise SystemExit("Providers kept failing during a probe — re-run to resume.")
    answer = oma_provider.strip_pedro_tags(res.get("reply") or "")
    expected = "; ".join(state["beats"][b]["memory"] for b in probe["expects"] if b in state["beats"]) \
        or "(none of the expected moments actually happened — a good answer says it has no such record)"
    verdict = judge.judge_probe(ledger_text(state), probe["question"], expected, answer)
    state["probes"].append({"after": spec["n"], "question": probe["question"], "answer": answer,
                            "pedro_was_given": dict(_last_prompt), "judgement": verdict})
    save(state)
    log(f"  probe '{probe['question'][:50]}': recall={verdict.get('recall', {}).get('score')} "
        f"fabrications={verdict.get('fabrications')}")


def main() -> int:
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {
        "user_id": None, "lessons": {}, "beats": {}, "probes": [], "snapshots": {}}
    if state["user_id"] is None:
        with SessionLocal() as db:
            u = User(email=f"adapt-eval-{int(time.time())}@coast.local", name=STUDENT["name"],
                     onboarding_completed=True)
            db.add(u)
            db.commit()
            state["user_id"] = u.id
        save(state)
    uid = state["user_id"]
    agent = ScenarioStudent(state)
    import claude_chat
    pedro = (f"Claude {claude_chat.PEDRO_MODEL}" if tutor.CHAT_PROVIDER == "anthropic"
             else f"Gemini {tutor.TUTOR_PROVIDERS['gemini']['model']}")
    state.setdefault("pedro_model", pedro)
    log(f"eval user {uid}; Pedro = {pedro}; judge = {judge.JUDGE_MODEL}; run dir {RUN_DIR}")

    for spec in LESSONS[:ARGS.lessons]:
        key = str(spec["n"])
        L = state["lessons"].setdefault(key, {"n": spec["n"], "name": spec["name"], "sections_log": []})
        if L.get("done"):
            continue
        if not L.get("setup"):
            info = copy_course(oma_provider.OMA_DB_PATH, *spec["src"], uid, spec["name"])
            L.update(setup=True, copy=info)
            save(state)
            log(f"LESSON {spec['n']} '{spec['name']}': {info['sections']} sections, {info['sources']} sources, "
                f"{info['content_items']} chunks, {info['concept_items']} concepts")
        L.setdefault("ledger_before", ledger_text(state))  # frozen when the lesson starts
        with SessionLocal() as db:
            sections = json.loads(db.query(database.CourseOutline)
                                  .filter_by(user_id=uid, folder_name=spec["name"]).one().outline_json)
        n_secs = min(len(sections), ARGS.max_sections or len(sections))
        agent.lesson_n = spec["n"]
        for idx in range(len(L["sections_log"]), n_secs):
            L["sections_log"].append(run_section(uid, agent, L, idx, sections[idx], state))
            save(state)
            judge_start(L, spec, state)
        judge_start(L, spec, state)
        logs = L["sections_log"]
        L.update(done=True, sections=len(logs), gates_passed=sum(s["gate_passed"] for s in logs),
                 gates_forced=sum(not s["gate_passed"] for s in logs))
        snap = memory_snapshot(uid, state)
        state["snapshots"][key] = {"memory": snap, "planted_in_memory": planted_in_memory(uid, state, snap)}
        save(state)
        log(f"  memory after lesson {key}: planted={state['snapshots'][key]['planted_in_memory']} "
            f"traits={len(snap['identity_traits'])} junk={snap['junk_traits']}")

        for probe in (p for p in PROBES if p["after"] == spec["n"]):
            if any(p["question"] == probe["question"] for p in state["probes"]):
                continue
            run_probe(uid, spec, probe, state)
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
