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


# ── Pedro on GPT-5.6 luna ─────────────────────────────────────────────────────
def _v2_turn(uid, context_type, luna, enabled=True):
    """One v2 turn (lesson or global chat) with a fixed request; `luna` stands in for OpenAI."""
    import openai_chat
    import pedro_context
    req = pedro_context.PedroRequest(
        system=[{"type": "text", "text": "Brief."}],
        messages=[{"role": "user", "content": [{"type": "text", "text": "What is utilisation?"}]}],
        fallback=[{"role": "system", "content": "Brief."}, {"role": "user", "content": "What is utilisation?"}],
        section_index=0 if context_type == "lesson" else None)
    builder = "lesson_request" if context_type == "lesson" else "open_request"
    saved = (tutor.PEDRO_CONTEXT, getattr(pedro_context, builder), openai_chat.stream_request, openai_chat.ENABLED)
    tutor.PEDRO_CONTEXT, openai_chat.stream_request, openai_chat.ENABLED = "v2", luna, enabled
    setattr(pedro_context, builder, lambda *a, **k: req)
    try:
        result = None
        kwargs = {"context_id": "Operations", "section_index": 0} if context_type == "lesson" else {}
        for _token, final in tutor.send_message_stream(uid, "What is utilisation?", None, context_type, **kwargs):
            if final is not None:
                result = final
        return result
    finally:
        tutor.PEDRO_CONTEXT, openai_chat.stream_request, openai_chat.ENABLED = saved[0], saved[2], saved[3]
        setattr(pedro_context, builder, saved[1])


def _luna_says(text):
    def fake(system, messages, **kw):
        fake.calls.append({"system": system, "messages": messages, **kw})
        yield text
    fake.calls = []
    return fake


def _luna_down(system, messages, **kw):
    import openai_chat
    raise openai_chat.OpenAIUnavailable("simulated outage")
    yield  # a generator, like the real one


def _saved_model(message_id):
    from database import ChatMessage, SessionLocal
    with SessionLocal() as db:
        return db.get(ChatMessage, message_id).model


def test_luna_writes_lesson_and_global_chat_turns():
    import openai_chat
    uid = make_student()
    make_course(uid, "Operations", [{"title": "Queues", "key_topics": ["utilisation"]}])
    for context_type in ("lesson", "global"):
        luna = _luna_says("Luna explains utilisation.")
        with claude(FakeAnthropic()) as fake:
            res = _v2_turn(uid, context_type, luna)
        assert res["reply"] == "Luna explains utilisation.", context_type
        assert _saved_model(res["message_id"]) == openai_chat.LUNA_MODEL
        assert luna.calls[0]["cache_key"] == f"pedro:{uid}:{res['conversation_id']}"
        assert not fake.calls, f"Claude must not also be billed for a {context_type} turn"


def test_claude_takes_the_turn_luna_cannot_answer():
    uid = make_student()
    make_course(uid, "Operations", [{"title": "Queues", "key_topics": ["utilisation"]}])
    with claude(FakeAnthropic()):
        res = _v2_turn(uid, "lesson", _luna_down)
    assert res["reply"] == "Claude says hi."
    assert _saved_model(res["message_id"]) == claude_chat.PEDRO_MODEL


def test_pedro_luna_off_puts_pedro_back_on_claude():
    uid = make_student()
    make_course(uid, "Operations", [{"title": "Queues", "key_topics": ["utilisation"]}])
    luna = _luna_says("Luna explains utilisation.")
    with claude(FakeAnthropic()):
        res = _v2_turn(uid, "lesson", luna, enabled=False)
    assert res["reply"] == "Claude says hi." and not luna.calls


def test_luna_receives_the_slides_without_claude_cache_marks():
    import openai_chat
    system = [{"type": "text", "text": "Brief", "cache_control": {"type": "ephemeral", "ttl": "1h"}}, {"type": "text", "text": "Frame"}]
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Page 24"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "What is A^2?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Let's look."}]},
        {"role": "user", "content": "It is 2."},
    ]
    out = openai_chat.to_openai(system, messages)
    assert out[0] == {"role": "developer", "content": "Brief\n\nFrame"}
    assert out[1]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}}
    assert out[2] == {"role": "assistant", "content": "Let's look."} and out[3] == {"role": "user", "content": "It is 2."}
    assert "cache_control" not in json.dumps(out)
