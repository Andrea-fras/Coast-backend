"""Query structured memory indexes across this student's courses, never other students."""
import json
import re
import sqlite3
from ..stores.db import connect_db
from ..course_identity import display_name

STOP = set('a an and are as at be been before can did do does explain for from had has have how i in is it last like me my of on or our please remember section semester so that the their them there these this to tutor us use was we were what when which with would you your'.split())

def recall_memories(db_path, user_id, query, exclude_namespace=None, limit=3):
    terms = list(dict.fromkeys(t for t in re.findall(r'[^\W_]+', str(query).lower()) if len(t)>2 and t not in STOP))[:16]
    if not terms:
        return []
    match = ' OR '.join('"'+term+'"' for term in terms)
    prefix = f'u{int(user_id)}__student__'
    out=[]
    seen=set()
    with connect_db(db_path) as conn:
        for store in ('pattern','episode'):
            table,fts=f'{store}_items',f'{store}_items_fts'
            try:
                rows=conn.execute(f"""SELECT p.id,p.namespace,p.content,p.store_specific,p.created_at
                    FROM {fts} JOIN {table} p ON p.id={fts}.id
                    WHERE {fts} MATCH ? AND p.namespace LIKE ? AND p.superseded_by IS NULL
                    ORDER BY rank LIMIT 24""",(match,prefix+'%')).fetchall()
            except sqlite3.OperationalError:
                continue
            for item_id,ns,content,raw,created in rows:
                if ns == exclude_namespace or item_id in seen:
                    continue
                data=json.loads(raw or '{}')
                if store=='pattern' and data.get('pattern_type')!='golden_moment':
                    continue
                if store=='episode' and data.get('episode_type') not in ('section_completed','section_evaluation'):
                    continue
                if not ns.startswith(prefix):
                    continue
                seen.add(item_id)
                out.append({'item_id':item_id,'folder':display_name(user_id,ns[len(prefix):]),
                    'text':content,'created_at':created,'section_index':data.get('section_index')})
                if len(out)>=limit:
                    return out
    return out
