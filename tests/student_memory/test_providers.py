"""Pedro on Claude (prompt caching, failover to the helper provider) and the
Claude-first section evaluator — verified with a fake Anthropic client, no network."""
import json
from contextlib import contextmanager
from types import SimpleNamespace

import claude_chat
import harness
from harness import chat, make_course, make_student, tutor


class FakeAnthropic:
    def __init__(self, reply="Claude says hi.", payload=None, stop_reason="end_turn"):
        self.calls, self.reply, self.payload, self.stop_reason = [], reply, payload, stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            events = [SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=self.reply)),
                      SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason=self.stop_reason))]
            return iter(events)
        text = json.dumps(self.payload or {})
        return SimpleNamespace(stop_reason=self.stop_reason, model=kw["model"],
                               content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))


@contextmanager
def claude(fake=None, pedro_on_claude=True):
    saved = (tutor.CHAT_PROVIDER, claude_chat._client, claude_chat.available)
    tutor.CHAT_PROVIDER = "anthropic" if pedro_on_claude else tutor.CHAT_PROVIDER
    claude_chat._client = fake
    claude_chat.available = lambda: True
    try:
        yield fake
    finally:
        tutor.CHAT_PROVIDER, claude_chat._client, claude_chat.available = saved


def test_pedro_on_claude_caches_fixed_instructions():
    uid = make_student()
    make_course(uid, "Operations", [{"title": "Queues", "key_topics": ["utilisation"]}])
    with claude(FakeAnthropic()) as fake:
        res = chat(uid, "What is utilisation?", "unused", context_id="Operations", section_index=0)
    assert res["reply"] == "Claude says hi."
    call = fake.calls[-1]
    assert call["model"] == claude_chat.PEDRO_MODEL and call["stream"] is True
    first, rest = call["system"][0], call["system"][1]
    assert first["text"] == tutor._pedro_static_prefix() and first["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in rest, "per-turn context must not be in the cached block"


def test_claude_failure_fails_over_to_helper_provider():
    uid = make_student()
    make_course(uid, "Operations", [{"title": "Queues", "key_topics": ["utilisation"]}])

    class Down(FakeAnthropic):
        def _create(self, **kw):
            raise claude_chat.ClaudeUnavailable("simulated outage")

    with claude(Down()):
        res = chat(uid, "What is utilisation?", "Fallback Pedro here.", context_id="Operations", section_index=0)
    assert res["reply"] == "Fallback Pedro here."


def test_evaluator_uses_claude_with_enforced_schema():
    refs = [{"concept_id": "c1", "concept_name": "Little's Law"}]
    payload = {"section_summary": "ok", "concepts": [{"concept_id": "c1", "concept_name": "Little's Law",
               "final_state": "mastered", "note": "n"}], "golden_moments": [], "open_questions": []}
    with claude(FakeAnthropic(payload=payload), pedro_on_claude=False) as fake:
        out = harness.ORIGINAL_EVALUATE("PEDRO: q\nSTUDENT: a", 0, "Queues", refs, require_model=True)
    call = fake.calls[-1]
    assert call["model"] == claude_chat.EVAL_MODEL
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert out["concepts"][0]["final_state"] == "mastered"


def test_evaluator_falls_back_when_claude_refuses():
    refs = [{"concept_id": "c1", "concept_name": "Little's Law"}]
    with claude(FakeAnthropic(stop_reason="refusal"), pedro_on_claude=False):
        out = harness.ORIGINAL_EVALUATE("PEDRO: q [ANSWER_CORRECT: Little's Law]\nSTUDENT: a", 0, "Queues", refs)
    assert out["concepts"], "no verdict after Claude refused"
