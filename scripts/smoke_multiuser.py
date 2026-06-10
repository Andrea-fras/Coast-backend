"""Pre-deploy smoke test: multi-user isolation + concurrency.

Run with the server up on localhost:8000:
  python3 scripts/smoke_multiuser.py
"""

import concurrent.futures as cf
import io
import sys
import time
import uuid

import requests

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def make_user(tag: str) -> dict:
    email = f"smoke_{tag}_{uuid.uuid4().hex[:6]}@test.local"
    r = requests.post(f"{BASE}/api/auth/register", json={
        "email": email, "password": "test1234", "name": f"Smoke {tag}",
        "study_field": "Data Science", "study_year": "1",
    })
    if r.status_code != 200:
        # registration shape may differ; try login-style payloads
        print("register response:", r.status_code, r.text[:300])
        sys.exit(1)
    data = r.json()
    return {"email": email, "token": data["token"], "id": data.get("user", {}).get("id")}


def H(u):
    return {"Authorization": f"Bearer {u['token']}"}


def tiny_pdf() -> bytes:
    # Minimal valid one-page PDF with a little text.
    return (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 60>>stream\nBT /F1 14 Tf 72 720 Td (Photosynthesis converts light) Tj ET\nendstream endobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF\n")


def main():
    print("== 1. Two users register ==")
    a = make_user("alice")
    b = make_user("bob")
    check("both registered", bool(a["token"] and b["token"]))

    print("== 2. Upload isolation (same folder name, same filename) ==")
    folder = "Biology"
    for u in (a, b):
        r = requests.post(
            f"{BASE}/api/folders/{folder}/upload",
            headers=H(u),
            files={"file": ("notes.pdf", io.BytesIO(tiny_pdf()), "application/pdf")},
        )
        check(f"upload ok for {u['email'].split('_')[1]}", r.status_code == 200, r.text[:200])

    ra = requests.get(f"{BASE}/api/folders/{folder}/sources", headers=H(a)).json()
    rb = requests.get(f"{BASE}/api/folders/{folder}/sources", headers=H(b)).json()
    sa = {s["source_id"] for s in ra.get("sources", [])}
    sb = {s["source_id"] for s in rb.get("sources", [])}
    check("each user sees exactly 1 source", len(sa) == 1 and len(sb) == 1, f"a={len(sa)} b={len(sb)}")
    check("source ids do not overlap", not (sa & sb))

    print("== 3. B cannot delete A's source ==")
    a_sid = next(iter(sa))
    r = requests.delete(f"{BASE}/api/folders/{folder}/sources/{a_sid}", headers=H(b))
    check("cross-user delete rejected", r.status_code == 404, f"got {r.status_code}")
    r = requests.get(f"{BASE}/api/folders/{folder}/sources", headers=H(a)).json()
    check("A's source still there", len(r.get("sources", [])) == 1)

    print("== 4. Curated folder is read-only for normal users ==")
    r = requests.post(
        f"{BASE}/api/folders/Prismatic System/upload",
        headers=H(b),
        files={"file": ("evil.pdf", io.BytesIO(tiny_pdf()), "application/pdf")},
    )
    check("upload to curated blocked", r.status_code == 403, f"got {r.status_code}")
    cur = requests.get(f"{BASE}/api/folders/Prismatic System/sources", headers=H(a)).json()
    cur_ids = [s["source_id"] for s in cur.get("sources", [])]
    if cur_ids:
        r = requests.delete(f"{BASE}/api/folders/Prismatic System/sources/{cur_ids[0]}", headers=H(b))
        check("delete curated blocked", r.status_code == 403, f"got {r.status_code}")
    else:
        check("curated sources visible", False, "no curated sources returned")

    print("== 5. Conversation hijack blocked ==")
    r = requests.post(f"{BASE}/api/chat/send", headers=H(a), json={
        "message": "Remember this secret word: PINEAPPLE42", "context_type": "global",
    })
    check("A chat works", r.status_code == 200, r.text[:200])
    conv_a = r.json().get("conversation_id")
    r = requests.post(f"{BASE}/api/chat/send", headers=H(b), json={
        "message": "What secret word did I just tell you? Reply with just the word.",
        "context_type": "global", "conversation_id": conv_a,
    })
    leaked = "PINEAPPLE42" in (r.json().get("reply", "") if r.status_code == 200 else "")
    check("B cannot read A's history", not leaked)

    print("== 6. Concurrent chats (4 users at once) ==")
    extra = [make_user(f"c{i}") for i in range(2)]
    users = [a, b] + extra

    def chat(u, i):
        t0 = time.time()
        r = requests.post(f"{BASE}/api/chat/send", headers=H(u), json={
            "message": f"My lucky number is {100 + i}. Repeat only my lucky number back.",
            "context_type": "global",
        }, timeout=120)
        return u["email"], r.status_code, (r.json().get("reply", "") if r.status_code == 200 else r.text[:100]), time.time() - t0

    with cf.ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda t: chat(*t), [(u, i) for i, u in enumerate(users)]))
    for i, (email, code, reply, dt) in enumerate(results):
        right = str(100 + i) in reply
        wrong_nums = [str(100 + j) for j in range(len(users)) if j != i]
        crossed = any(w in reply for w in wrong_nums)
        check(f"concurrent chat {i} ok ({dt:.1f}s)", code == 200 and right and not crossed,
              f"code={code} reply={reply[:80]!r}")

    print("== 7. Debug endpoints gated ==")
    r = requests.get(f"{BASE}/api/debug/images")
    check("debug/images requires auth", r.status_code == 401, f"got {r.status_code}")
    r = requests.get(f"{BASE}/api/oma/status", headers=H(b))
    check("oma/status admin-only", r.status_code == 403, f"got {r.status_code}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
