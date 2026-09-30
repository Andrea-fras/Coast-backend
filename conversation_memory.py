"""Reuse an earlier summary and summarize only newly accumulated older turns."""
from sqlalchemy.dialects.sqlite import insert
from database import SessionLocal, ChatMessage, ConversationDigest


def context(user_id, conversation_id, summarize, threshold=12, keep_recent=6):
    with SessionLocal() as db:
        digest = db.get(ConversationDigest, (user_id, conversation_id))
        through = digest.through_message_id if digest else 0
        summary = digest.summary if digest else None
        pending = db.query(ChatMessage).filter(ChatMessage.user_id == user_id,
            ChatMessage.conversation_id == conversation_id, ChatMessage.id > through).order_by(ChatMessage.id).all()
    if len(pending) > threshold:
        older = pending[:-keep_recent]
        updated = summarize(older, summary)
        if updated:
            new_through = older[-1].id
            with SessionLocal() as db:
                statement = insert(ConversationDigest).values(user_id=user_id, conversation_id=conversation_id,
                    through_message_id=new_through, summary=updated)
                db.execute(statement.on_conflict_do_update(index_elements=['user_id','conversation_id'],
                    set_={'through_message_id':new_through,'summary':updated},
                    where=ConversationDigest.through_message_id <= through))
                db.commit()
            return updated, pending[-keep_recent:]
    # Preserve every unsummarized turn when reusing a summary. A failed summary
    # leaves the watermark untouched, so a later call can retry without a gap.
    limit = max(20, threshold)
    if len(pending) > limit:
        summary = (summary or '') + '\nSome intervening turns could not be summarized; do not infer their contents.'
    return summary, pending[-limit:]
