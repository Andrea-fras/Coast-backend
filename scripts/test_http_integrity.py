#!/usr/bin/env python3
"""Exercise real routes with isolated SQLite and no startup jobs or LLM calls."""
import os,sys,tempfile,json,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TMP=tempfile.TemporaryDirectory(prefix='coast-http-')
for key,name in {'DATABASE_PATH':'app.db','OMA_DB_PATH':'oma.db','CHROMA_PATH':'chroma',
                 'GENERATED_DIR':'generated','FOLDER_UPLOADS_DIR':'sources','OMA_IMAGE_DIR':'images'}.items():
    os.environ[key]=str(Path(TMP.name)/name)
for key in ('OPENAI_API_KEY','GEMINI_API_KEY','ANTHROPIC_API_KEY'):
    os.environ.pop(key,None)
import dotenv
dotenv.load_dotenv=lambda *a,**k: None
from fastapi.testclient import TestClient
import server
from database import Base,engine,SessionLocal,User,CourseOutline,SourceImage,LearningJob,SectionRewardClaim,UserMapState
from auth import create_access_token
import placement,lesson,map_world
client=TestClient(server.app)  # No lifespan: do not ingest curated courses.

class HttpIntegrity(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine);Base.metadata.create_all(engine)
        with SessionLocal() as db:
            db.add_all([User(id=1,email='one@example.invalid',name='One'),User(id=2,email='two@example.invalid',name='Two')])
            db.add(CourseOutline(user_id=1,folder_name='Physics',current_section=0,total_sections=3,
                outline_json=json.dumps([{'title':'A'},{'title':'B'},{'title':'C'}])))
            db.add(SourceImage(id=1,user_id=1,folder_name='Physics',source_id='fixture',page_number=1,
                image_path=str(Path(TMP.name)/'fixture.png')))
            db.commit()
        Path(TMP.name,'fixture.png').write_bytes(b'fixture')
        map_world.invalidate_map_cache(1)
        self.h={'Authorization':'Bearer '+create_access_token(1,'one@example.invalid')}
        self.other={'Authorization':'Bearer '+create_access_token(2,'two@example.invalid')}

    def test_notes_conflict_and_retry(self):
        url='/api/folders/Physics/lesson-notes'
        initial=client.get(url,headers=self.h).json()
        saved=client.put(url,headers=self.h,json={'content_html':'<p>First</p>','revision':initial['revision']})
        self.assertEqual(saved.status_code,200,saved.text)
        replay=client.put(url,headers=self.h,json={'content_html':'<p>First</p>','revision':initial['revision']})
        self.assertEqual(replay.status_code,200)
        conflict=client.put(url,headers=self.h,json={'content_html':'<p>Stale</p>','revision':initial['revision']})
        self.assertEqual(conflict.status_code,409)
        self.assertEqual(client.get(url,headers=self.h).json()['content_html'],'<p>First</p>')
        self.assertEqual(client.get(url,headers=self.other).json()['content_html'],'')

    def test_source_images_require_owner_or_scoped_token(self):
        url='/api/source-images/1'
        self.assertEqual(client.get(url).status_code,401)
        self.assertEqual(client.get(url,headers=self.other).status_code,404)
        self.assertEqual(client.get(url,headers=self.h).status_code,200)
        access=client.get('/api/image-access',headers=self.h).json()['access']
        self.assertEqual(client.get(url,params={'access':access}).status_code,200)
        self.assertEqual(client.get('/api/lessons/summary',headers={'Authorization':'Bearer '+access}).status_code,401)

    def test_test_out_rejects_forgery_accepts_and_replays_real_pass(self):
        url='/api/folders/Physics/lesson/test-out'
        self.assertEqual(client.post(url,headers=self.h,json={'target_section':2}).status_code,400)
        placement.begin(1,'Physics',2,'fixture');placement.record_pass(1,'fixture')
        body={'target_section':2,'conversation_id':'fixture'}
        for _ in range(2):
            response=client.post(url,headers=self.h,json=body)
            self.assertEqual(response.status_code,200,response.text)
        with SessionLocal() as db:
            self.assertEqual(db.query(LearningJob).count(),2)
            self.assertEqual(db.query(CourseOutline).first().current_section,2)

    def test_verification_records_final_section_without_advance(self):
        lesson.mark_section_verified(1,'Physics',2)
        lesson.mark_section_verified(1,'Physics',2)
        with SessionLocal() as db:
            self.assertEqual(db.query(LearningJob).count(),1)
            payload=json.loads(db.query(LearningJob).first().payload_json)
            self.assertTrue(payload['lesson_complete'])

    def test_startup_schedules_recovery_without_import_shadowing(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch('server.init_db'))
            stack.enter_context(patch('coast_content_oma.course_identity.initialize'))
            stack.enter_context(patch('learning_jobs.start'))
            stack.enter_context(patch('server.load_papers_from_json'))
            stack.enter_context(patch('paper_scanner.load_scanned_into_db'))
            thread = stack.enter_context(patch('server.threading.Thread'))
            server.on_startup()
            self.assertEqual(thread.call_count, 2)
            self.assertEqual(thread.return_value.start.call_count, 2)

    def test_map_reward_and_compact_snapshot(self):
        # Avoid model-derived legacy mastery; fixture has only authoritative claims.
        with patch.object(lesson,'get_section_mastery_list',return_value=[]):
            first=map_world.claim_section_reward(1,'Physics',0,section_title='A')
            again=map_world.claim_section_reward(1,'Physics',0,section_title='A')
            self.assertEqual(first['total_xp'],again['total_xp'])
            state=map_world.get_map_state(1,compact=True)
            self.assertEqual(state['sections_mastered'],1)
            self.assertIn('section_catalog',state)
            with patch.object(map_world,'_build_map_state',side_effect=AssertionError('unnecessary rebuild')):
                map_world.get_map_state(1,compact=True)
                map_world._map_cache.clear()  # Simulate a process restart or LRU eviction.
                self.assertEqual(map_world.get_map_state(1,compact=True)['tile_sections'],state['tile_sections'])
            with SessionLocal() as db:
                self.assertEqual(db.query(SectionRewardClaim).count(),1)

    def test_treasure_question_is_pinned_and_wrong_attempt_consumed_once(self):
        import treasure
        from database import TreasureChestOpen
        chest=treasure.TREASURE_CHESTS[0];cid=chest['id']
        challenge={'chest_id':cid,'chest_name':chest['name'],'challenge_id':'fixture',
            'concept_name':'Energy','folder':'Physics','section':'A','question':'Explain energy','model_answer':'Conserved quantity'}
        with patch.object(map_world,'get_map_state',return_value={'reveal_radius':10}), \
             patch.object(map_world,'_unlocked_cells',return_value={(chest['x'],chest['y'])}), \
             patch.object(treasure,'_build_challenge',return_value=challenge) as build:
            first=treasure.build_quiz(1,cid)
            self.assertNotIn('model_answer',first)
            treasure.build_quiz(1,cid)
            self.assertEqual(build.call_count,1)
            self.assertIn('error',treasure.build_quiz(1,'0,0'))
            with patch.object(treasure,'_grade_answer',return_value={'score':0,'is_correct':False,'feedback':'Try a conservation example.'}):
                result=treasure.complete_treasure(1,cid,'A sufficiently long wrong answer')
                self.assertTrue(result['consumed'])
                self.assertEqual(result['xp_gained'],0)
                self.assertEqual(treasure.complete_treasure(1,cid,'Another answer')['error'],'already_opened')
        with SessionLocal() as db: self.assertEqual(db.query(TreasureChestOpen).count(),1)

    def test_treasure_provider_failure_does_not_consume_chest(self):
        import treasure
        from database import TreasureChallenge,TreasureChestOpen
        cid=treasure.TREASURE_CHESTS[0]['id']
        with SessionLocal() as db:
            db.add(TreasureChallenge(user_id=1,chest_id=cid,challenge_json=json.dumps({
                'question':'Q','model_answer':'A','concept_name':'Energy'})));db.commit()
        with patch.object(treasure,'_grade_answer',return_value={'error':'grading_unavailable','feedback':'Retry'}):
            self.assertEqual(treasure.complete_treasure(1,cid,'Valid student response')['error'],'grading_unavailable')
        with SessionLocal() as db: self.assertEqual(db.query(TreasureChestOpen).count(),0)

    def test_resume_restores_latest_owned_conversation_and_respects_regeneration(self):
        from database import ChatMessage,CourseChatEpoch
        with SessionLocal() as db:
            for uid,conv,content in [(1,'old','Old attempt'),(2,'private','Other student'),(1,'latest','Saved explanation')]:
                db.add(ChatMessage(user_id=uid,conversation_id=conv,role='pedro',content=content,
                    context_type='lesson',context_id='Physics',section_index=0))
            db.commit()
        url='/api/folders/Physics/section-chat/0?resume=true'
        response=client.get(url,headers=self.h).json()
        self.assertEqual(response['conversation_id'],'latest')
        self.assertEqual([m['content'] for m in response['messages']],['Saved explanation'])
        with SessionLocal() as db:
            last=db.query(ChatMessage).order_by(ChatMessage.id.desc()).first().id
            db.add(CourseChatEpoch(user_id=1,folder_name='Physics',through_message_id=last));db.commit()
        self.assertEqual(client.get(url,headers=self.h).json()['messages'],[])
        self.assertEqual(len(client.get('/api/folders/Physics/section-chat/0',headers=self.h).json()['messages']),2)

    def test_unprepared_section_is_rejected_before_chat_starts(self):
        with SessionLocal() as db:
            outline=db.query(CourseOutline).filter_by(user_id=1,folder_name='Physics').first()
            outline.outline_json=json.dumps([{'title':'A','preparation_version':1,'source_refs':[{'source_id':'missing','pages':[1],'sha256':'old'}]}]);db.commit()
        with patch('tutor.send_message_stream') as chat:
            response=client.post('/api/chat/stream',headers=self.h,json={'context_type':'lesson','context_id':'Physics','section_index':0,'message':'Teach me'})
            self.assertEqual(response.status_code,409)
            chat.assert_not_called()
        from database import ChatMessage
        with SessionLocal() as db: self.assertEqual(db.query(ChatMessage).count(),0)

    def test_test_out_waits_only_for_the_first_section_it_checks(self):
        with SessionLocal() as db:
            outline=db.query(CourseOutline).filter_by(user_id=1,folder_name='Physics').first()
            outline.outline_json=json.dumps([{'title':title,'preparation_version':1,'source_refs':[]} for title in ('A','B','C')])
            db.commit()
        from coast_content_oma import progressive
        from database import ChatMessage
        body={'context_type':'test_out','context_id':'Physics','section_index':2,'message':'Test me'}
        for endpoint,method in (('stream','send_message_stream'),('send','send_message')):
            with patch.object(progressive,'status_for_section',return_value={'ready':False}) as readiness, patch('tutor.'+method) as chat:
                response=client.post('/api/chat/'+endpoint,headers=self.h,json=body)
                self.assertEqual(response.status_code,409,response.text)
                self.assertEqual([c.args[2]['title'] for c in readiness.call_args_list],['A'])
                chat.assert_not_called()
        with patch.object(progressive,'status_for_section',return_value={'ready':True}) as readiness:
            progressive.assert_chat_ready(1,'Physics',2,test_out=True)
            self.assertEqual([c.args[2]['title'] for c in readiness.call_args_list],['A'])
        with SessionLocal() as db: self.assertEqual(db.query(ChatMessage).count(),0)

    def test_unused_manim_route_is_not_public(self):
        self.assertFalse(any(getattr(route, 'path', '') == '/api/visualize' for route in server.app.routes))

if __name__=='__main__': unittest.main(verbosity=2)
