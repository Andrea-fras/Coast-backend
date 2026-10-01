"""Progressive preparation regressions; isolated databases and mocked provider calls."""
import sys,json,tempfile,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.test_launch_integrity as fixture
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from coast_content_oma import progressive as p
from coast_content_oma.ingestion import IngestionPipeline,_prioritized_batches
from coast_content_oma.stores.content import ContentStore
from coast_content_oma.stores.image import ImageStore
from coast_content_oma.stores.concept import ConceptStore
from coast_content_oma.stores.base import MemoryItem
from coast_content_oma.stores.db import connect_db
from database import SessionLocal,FolderSource,CourseOutline,LearningJob
import learning_jobs

class ProgressiveTests(unittest.TestCase):
    setUp=fixture.LaunchIntegrity.setUp
    def stores(self,path):
        return SimpleNamespace(content=ContentStore(path),images=ImageStore(path),concept=ConceptStore(path))
    def source(self,sid,count):
        return SimpleNamespace(source_id=sid,title=sid,filename=sid+'.pdf',page_count=count,file_path=sid)
    def manifest(self,count):
        return {'sha256':'fixture','pages':[{'page_number':n,'text':('Topic '+str(n)+' details ')*20,'images':[]} for n in range(1,count+1)]}
    def test_credit_exhaustion_pauses_embeddings_but_transient_limits_do_not(self):
        from coast_content_oma import embedding_health as health
        with patch.object(health,'_retry_at',0),patch.object(health.time,'monotonic',return_value=10):
            self.assertTrue(health.available())
            self.assertFalse(health.note_error(Exception('429 rate limit')))
            self.assertTrue(health.available())
            self.assertTrue(health.note_error(Exception('credit_balance_exhausted')))
            self.assertFalse(health.available())
        with patch.object(health,'_retry_at',310),patch.object(health.time,'monotonic',return_value=311):
            self.assertTrue(health.available())

    def test_overview_represents_all_sources_and_pages(self):
        sources=[self.source('a',65),self.source('b',16)]
        with patch.object(p,'manifest',side_effect=lambda s:self.manifest(s.page_count)):
            text,units=p.overview(sources)
        self.assertIn('p.65',text);self.assertIn('UNIT b:9-16',text)
        self.assertLess(len(text),70000)
        self.assertEqual(sum(len(u['pages']) for u in units.values()),81)
    def test_large_lecture_set_preserves_every_page_within_budget(self):
        sources=[self.source('lecture'+str(i),count) for i,count in enumerate([44,53,29,55,58,58,142])]
        manifests={s.source_id:self.manifest(s.page_count) for s in sources}
        # Include short, blank, and very dense slides rather than assuming equal sizes.
        for data in manifests.values():
            for row in data['pages']:
                row['text'] = 'Short slide' if row['page_number']%3==0 else ('Dense topic details '*500)
        with patch.object(p,'manifest',side_effect=lambda s:manifests[s.source_id]):
            text,units=p.overview(sources)
        self.assertLessEqual(len(text),70000)
        self.assertEqual(sum(len(u['pages']) for u in units.values()),439)
        for source in sources:
            pages=[page for unit in units.values() if unit['source_id']==source.source_id for page in unit['pages']]
            self.assertEqual(pages,list(range(1,source.page_count+1)))
        self.assertIn('Short slide',text)
        self.assertEqual(sum(line.startswith('p.') for line in text.splitlines()),439)

    def test_short_pages_leave_budget_for_dense_pages(self):
        source=self.source('lecture',2)
        data=self.manifest(2)
        data['pages'][0]['text']='Short'
        data['pages'][1]['text']='D'*2000
        with patch.object(p,'manifest',return_value=data):
            text,_=p.overview([source],max_chars=900)
        self.assertEqual(len(text),900)
        self.assertIn('p.1 (0 images): Short',text)
        self.assertGreater(text.count('D'),500)

    def test_missing_units_are_attached_and_invented_units_dropped(self):
        units={'a:1-8':{'source_id':'a','source_title':'A','pages':list(range(1,9))},'b:1-8':{'source_id':'b','source_title':'B','pages':list(range(1,9))}}
        # A section citing only invented units is still unusable.
        with self.assertRaises(ValueError): p.bind_sections([{'title':'test','source_units':['invented']}],units)
        # Every uploaded page is still taught: the forgotten unit is attached, the invented one dropped.
        grouped=p.bind_sections([{'title':'one','source_units':['a:2']}],{'a:1-8':units['a:1-8']})
        self.assertEqual(grouped[0]['source_units'],['a:1-8'])  # a page inside a grouped unit resolves to it
        borrowed=p.bind_sections([{'title':'one','source_units':['a:1-8']},{'title':'two','source_units':['invented']}],units)
        self.assertEqual(borrowed[1]['source_units'][:1],['a:1-8'])  # a section with only invented refs teaches its neighbour's pages
        sections=p.bind_sections([{'title':'test','source_units':['a:1-8','invented']}],units)
        self.assertEqual([r['source_id'] for r in sections[0]['source_refs']],['a','b'])
        sections=p.bind_sections([{'title':'test','source_units':list(units)}],units)
        self.assertEqual(len(sections[0]['source_refs']),2)
        # After a failed repair, unusable references fall back to pages in source order.
        spread=p.bind_sections([{'title':'one','source_units':['x']},{'title':'two'},{'title':'three','source_units':['y']}],units,spread=True)
        self.assertEqual([s['source_units'] for s in spread],[['a:1-8'],['a:1-8'],['b:1-8']])
    def test_scheduler_reprioritizes_unsent_work(self):
        ranks={0:0,1:1,2:2}
        results=[]
        for job,result in _prioritized_batches([0,1,2],lambda n:n,1,lambda n:ranks[n]):
            results.append(result)
            if job==0: ranks[2]=-1
        self.assertEqual(results,[0,2,1])
    def test_priorities_use_source_and_page(self):
        p.set_priority('u1__fixture',[{'source_refs':[{'source_id':'B','pages':[1]}]},{'source_refs':[{'source_id':'A','pages':[1]}]}])
        self.assertLess(p.page_rank('u1__fixture','doc_B',1),p.page_rank('u1__fixture','doc_A',1))
    def test_first_section_ready_while_remainder_still_classifies(self):
        with tempfile.TemporaryDirectory() as d:
            orch=self.stores(Path(d)/'oma.db');ns='u1__fixture'
            pipeline=IngestionPipeline(orch.concept,orch.content,orch.images,Path(d)/'images',describe_images=False,classify_workers=1)
            pages=self.manifest(16)['pages'];release=threading.Event();later=threading.Event();finished=[]
            def classify(batch):
                if batch[0]['page_number']>8: later.set();release.wait(5)
                return [{'page_number':page['page_number'],'content_types':['definition'],'concepts':[]} for page in batch]
            with patch('coast_content_oma.ingestion.extract_pages',return_value=pages),patch.object(pipeline,'_classify_batch',side_effect=classify):
                thread=threading.Thread(target=lambda:finished.append(pipeline.ingest_folder(ns,[Path(d)/'lecture.pdf'],source_ids={'lecture.pdf':'srcA'},defer_concepts=True)))
                thread.start()
                try:
                    self.assertTrue(later.wait(3))
                    status=p.section_status(orch,ns,{'source_refs':[{'source_id':'srcA','pages':list(range(1,9))}]})
                    self.assertTrue(status['ready']);self.assertTrue(thread.is_alive())
                    self.assertFalse(p.section_status(orch,ns,{'source_refs':[{'source_id':'srcA','pages':[9]}]})['ready'])
                finally:release.set();thread.join(5)
            self.assertEqual(finished[0].content_items,16);self.assertEqual(finished[0].errors,[])
    def test_missing_diagram_blocks_only_its_assigned_section(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            orch=self.stores(Path(d)/'oma.db');ns='u1__fixture';sid='doc_a'
            pages=self.manifest(2)['pages']
            p.register_pages(orch.content.db_path,ns,sid,pages,[{'page_number':2,'file_path':'diagram.png'}])
            p.mark_text(orch.content.db_path,ns,sid,pages)
            self.assertTrue(p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[1]}]})['ready'])
            self.assertFalse(p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[2]}]})['ready'])
            iid='ima_'+hashlib.sha256(f'{ns}:{sid}:diagram.png'.encode()).hexdigest()
            orch.images._insert(MemoryItem(id=iid,namespace=ns,store='image',content='A labeled diagram',store_specific={}))
            self.assertTrue(p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[2]}]})['ready'])
            self.assertFalse(p.section_status(orch,'u2__fixture',{'source_refs':[{'source_id':'a','pages':[2]}]})['ready'])
    def test_vision_batches_publish_before_remaining_diagrams_finish(self):
        with tempfile.TemporaryDirectory() as d:
            orch=self.stores(Path(d)/'oma.db');ns='u1__fixture'
            pipeline=IngestionPipeline(orch.concept,orch.content,orch.images,Path(d)/'images',classify_workers=1,vision_workers=1)
            pipeline.VISION_BATCH_SIZE=1
            saved=[]
            for page in (1,2):
                path=Path(d)/f'{page}.png';path.write_bytes(b'fixture')
                saved.append({'page_number':page,'file_path':str(path),'width':100,'height':100})
            release=threading.Event();later=threading.Event();calls=[];results=[]
            def describe(batch):
                calls.append(1)
                if len(calls)==2: later.set();release.wait(5)
                return [{'description':'A useful diagram','image_type':'diagram','concepts':[]}]
            with patch('coast_content_oma.ingestion.extract_pages',return_value=self.manifest(2)['pages']),patch.object(pipeline,'_save_images_to_disk',return_value=saved),patch.object(pipeline,'_classify_batch',return_value=[]),patch('coast_content_oma.llm.describe_images_batch',side_effect=describe):
                thread=threading.Thread(target=lambda:results.append(pipeline.ingest_folder(ns,[Path(d)/'a.pdf'],source_ids={'a.pdf':'a'},defer_concepts=True)))
                thread.start()
                try:
                    self.assertTrue(later.wait(3))
                    import time
                    deadline=time.monotonic()+2
                    status={'ready':False}
                    while time.monotonic()<deadline and not status['ready']:
                        status=p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[1]}]})
                        if not status['ready']: time.sleep(.01)
                    self.assertTrue(status['ready']);self.assertTrue(thread.is_alive())
                    self.assertFalse(p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[2]}]})['ready'])
                finally:release.set();thread.join(5)
                before=len(calls)
                again=pipeline.ingest_folder(ns,[Path(d)/'a.pdf'],source_ids={'a.pdf':'a'},defer_concepts=True)
                self.assertEqual(len(calls),before)  # completed diagrams are reused on replay
                self.assertEqual(again.content_items,2)
            self.assertEqual(results[0].image_items,2)

    def test_a_figure_heavy_file_is_read_without_describing_every_figure(self):
        """Past INLINE_FIGURES, a file's figures go to the background sweep (the student's section
        first) instead of holding up the file, and the next one, until all are described."""
        with tempfile.TemporaryDirectory() as d:
            orch=self.stores(Path(d)/'oma.db');ns='u1__inline'
            pipeline=IngestionPipeline(orch.concept,orch.content,orch.images,Path(d)/'images',classify_workers=1,vision_workers=1)
            pipeline.VISION_BATCH_SIZE=1;pipeline.INLINE_FIGURES=1
            saved=[]
            for page in (1,2):
                path=Path(d)/f'{page}.png';path.write_bytes(b'fixture')
                saved.append({'page_number':page,'file_path':str(path),'width':100,'height':100})
            p.set_priority(ns,[{'source_refs':[{'source_id':'a','pages':[2]}]}],0)  # they're on the section with page 2
            described=[]
            def describe(batch):
                described.append(1)
                return [{'description':'A useful diagram','image_type':'diagram','concepts':[]}]
            with patch('coast_content_oma.ingestion.extract_pages',return_value=self.manifest(2)['pages']),patch.object(pipeline,'_save_images_to_disk',return_value=saved),patch.object(pipeline,'_classify_batch',return_value=[]),patch('coast_content_oma.llm.describe_images_batch',side_effect=describe):
                stats=pipeline.ingest_folder(ns,[Path(d)/'a.pdf'],source_ids={'a.pdf':'a'},defer_concepts=True)
                self.assertEqual(len(described),1)
                self.assertEqual(stats.figures_pending,1)
                self.assertTrue(p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[2]}]})['ready'])
                self.assertFalse(p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[1]}]})['ready'])
                pipeline.describe_pending_images(ns)
            self.assertTrue(p.section_status(orch,ns,{'source_refs':[{'source_id':'a','pages':[1]}]})['ready'])

    def test_outline_billing_failure_is_actionable_and_fallback_still_works(self):
        import os,lesson
        from unittest.mock import MagicMock
        gemini=MagicMock()
        gemini.models.generate_content.side_effect=RuntimeError('429 Your prepayment credits are depleted')
        fallback=MagicMock()
        fallback.chat.completions.create.side_effect=RuntimeError('credit_balance_exhausted')
        with patch.dict(os.environ,{'GEMINI_API_KEY':'fixture','OPENAI_API_KEY':'fixture'}), patch.object(lesson,'_request_gemini_outline',side_effect=gemini.models.generate_content), patch('openai.OpenAI',return_value=fallback):
            with self.assertRaisesRegex(lesson.OutlineProviderError,'Gemini.*OpenAI'):
                lesson._call_llm_for_outline('system','context')
            # While the cooldown lasts, providers out of credits are skipped without a request.
            import provider_capacity
            calls=fallback.chat.completions.create.call_count
            with self.assertRaisesRegex(lesson.OutlineProviderError,'credits'):
                lesson._call_llm_for_outline('system','context')
            self.assertEqual(fallback.chat.completions.create.call_count,calls)
            provider_capacity._down_until.clear()  # the cooldown ends after the top-up
            fallback.chat.completions.create.side_effect=None
            fallback.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[{"title":"Fallback"}]'))])
            self.assertEqual(lesson._call_llm_for_outline('system','context'),[{'title':'Fallback'}])

    def test_gemini3_uses_low_thinking_and_bounded_json_output(self):
        import os,lesson
        from unittest.mock import Mock
        response=Mock(is_error=False);response.json.return_value={'candidates':[]}
        with patch.dict(os.environ,{'GEMINI_API_KEY':'fixture','GEMINI_OUTLINE_MODEL':'gemini-3.1-pro-preview'}),patch('httpx.post',return_value=response) as request:
            lesson._request_gemini_outline('system','context',14536)
        config=request.call_args.kwargs['json']['generationConfig']
        self.assertEqual(config['thinkingConfig'],{'thinkingLevel':'low'})
        self.assertEqual(config['maxOutputTokens'],14536)
        self.assertEqual(config['responseMimeType'],'application/json')

    def test_invalid_gemini_json_tries_fallback_with_large_output_budget(self):
        import os,lesson
        from unittest.mock import MagicMock
        gemini=MagicMock()
        gemini.models.generate_content.return_value={'candidates':[{'content':{'parts':[{'text':'[{"title":"cut off'}]},'finishReason':'MAX_TOKENS'}]}
        fallback=MagicMock()
        fallback.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[{"title":"Recovered"}]'))])
        with patch.dict(os.environ,{'GEMINI_API_KEY':'fixture','OPENAI_API_KEY':'fixture'}), patch.object(lesson,'_request_gemini_outline',side_effect=gemini.models.generate_content), patch('openai.OpenAI',return_value=fallback):
            result=lesson._call_llm_for_outline('system','\nUNIT x'*58)
        self.assertEqual(result,[{'title':'Recovered'}])
        self.assertGreater(gemini.models.generate_content.call_args.args[2],8192)
        self.assertEqual(fallback.chat.completions.create.call_args.kwargs['model'],'gpt-4o-mini')

    def test_credit_failures_do_not_retry_as_transient_rate_limits(self):
        from coast_content_oma import llm
        from unittest.mock import Mock
        operation=Mock(side_effect=RuntimeError('429 prepayment credits are depleted'))
        with patch.object(llm.time,'sleep') as sleep:
            self.assertIsNone(llm._retry_on_rate_limit(operation))
            operation.assert_called_once();sleep.assert_not_called()

    def test_explicit_source_retry_keeps_finished_work_and_live_leases(self):
        import time
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pdf';path.write_bytes(b'fixture')
            with SessionLocal() as db:
                for uid,sid,status in [(1,'failed','FAILED'),(1,'done','CONTENT_INDEXED'),(1,'active','INGESTING'),(2,'private','FAILED')]:
                    source=FolderSource(user_id=uid,folder_name='Physics',source_id=sid,title=sid,filename='a.pdf',source_type='pdf',page_count=1,raw_text='fixture',file_path=str(path),oma_ingest_status=status)
                    db.add(source);db.flush()
                    jid=learning_jobs.enqueue_source(db,source)
                    job=db.get(LearningJob,jid);job.status='running' if sid=='active' else 'failed'
                    job.attempts=8;job.lease_until=time.time()+120 if sid=='active' else 0
                db.commit()
            self.assertEqual(learning_jobs.retry_sources(1,'Physics'),['failed'])
            with SessionLocal() as db:
                jobs={json.loads(j.payload_json)['source_id']:j for j in db.query(LearningJob)}
                self.assertEqual(jobs['failed'].status,'queued');self.assertEqual(jobs['failed'].attempts,0)
                self.assertEqual(jobs['active'].status,'running');self.assertEqual(jobs['private'].status,'failed')
                self.assertEqual(db.query(FolderSource).filter_by(source_id='done').first().oma_ingest_status,'CONTENT_INDEXED')

    def test_legacy_vision_sweeper_does_not_duplicate_source_job_images(self):
        with tempfile.TemporaryDirectory() as directory:
            orch=self.stores(Path(directory)/'oma.db');ns='u1__fixture'
            pipeline=IngestionPipeline(orch.concept,orch.content,orch.images,Path(directory)/'images')
            p.register_pages(orch.content.db_path,ns,'doc_a',self.manifest(1)['pages'],[])
            orch.images._insert(MemoryItem(id='pending',namespace=ns,store='image',source_doc_id='doc_a',content='',store_specific={'_pending_vision':True,'page_number':1}))
            # While its source job is still running, the sweeper leaves its figures alone…
            with patch.object(pipeline,'_describe_saved_images_batched') as describe:
                result=pipeline.describe_pending_images(ns,skip_doc_ids={'doc_a'})
            self.assertEqual(result['pending_total'],0);describe.assert_not_called()
            # …and once the job has finished, figures it could not describe are swept.
            done={'page_number':1,'description':'A graph','image_type':'diagram','concepts':[]}
            with patch.object(pipeline,'_describe_saved_images_batched',return_value=[done]) as describe:
                orch.images._insert(MemoryItem(id='pending',namespace=ns,store='image',source_doc_id='doc_a',content='',
                    store_specific={'_pending_vision':True,'page_number':1,'file_path':__file__}))
                done['file_path']=__file__
                result=pipeline.describe_pending_images(ns)
            self.assertEqual(result['described'],1)
            self.assertEqual(orch.images.get('pending').content,'A graph')

    def test_figures_that_keep_failing_stop_blocking_their_section(self):
        with tempfile.TemporaryDirectory() as directory:
            orch=self.stores(Path(directory)/'oma.db');ns='u1__fixture'
            pipeline=IngestionPipeline(orch.concept,orch.content,orch.images,Path(directory)/'images')
            orch.images._insert(MemoryItem(id='stuck',namespace=ns,store='image',source_doc_id='doc_a',content='',
                store_specific={'_pending_vision':True,'page_number':1,'file_path':__file__}))
            failing={'page_number':1,'file_path':__file__,'description':'','_pending_vision':True}
            with patch.object(pipeline,'_describe_saved_images_batched',return_value=[failing]) as describe, \
                    patch('coast_content_oma.ingestion.time.sleep'):
                pipeline.describe_pending_images(ns)
            self.assertEqual(describe.call_count,IngestionPipeline.VISION_ATTEMPTS)
            item=orch.images.get('stuck')
            self.assertFalse(item.store_specific.get('_pending_vision'))
            self.assertEqual(item.content,'(figure — description unavailable)')

    def test_the_current_sections_figures_are_saved_before_the_rest_are_described(self):
        """Figures are described in roadmap order and saved chunk by chunk, so the section the
        student is on stops waiting long before the whole folder's sweep ends."""
        with tempfile.TemporaryDirectory() as directory:
            orch=self.stores(Path(directory)/'oma.db');ns='u1__chunked'
            pipeline=IngestionPipeline(orch.concept,orch.content,orch.images,Path(directory)/'images')
            chunk=pipeline.VISION_BATCH_SIZE*pipeline.vision_workers
            for n in range(chunk+2):  # one figure on each page of doc_a
                f=Path(directory)/f'f{n}.png';f.write_bytes(b'x')
                orch.images._insert(MemoryItem(id=f'img{n}',namespace=ns,store='image',source_doc_id='doc_a',content='',
                    store_specific={'_pending_vision':True,'page_number':n+1,'file_path':str(f)}))
            p.set_priority(ns,[{'source_refs':[{'source_id':'a','pages':[chunk+1,chunk+2]}]}],0)  # the section they're on
            seen=[]
            def describe(saved):
                seen.append([s['page_number'] for s in saved])
                if len(seen)==2:  # the first chunk is already saved while the rest is still being described
                    self.assertFalse(orch.images.get(f'img{chunk}').store_specific.get('_pending_vision'))
                return [{**s,'description':'fig','image_type':'diagram','concepts':[]} for s in saved]
            with patch.object(pipeline,'_describe_saved_images_batched',side_effect=describe):
                result=pipeline.describe_pending_images(ns)
            self.assertEqual(seen[0][:2],[chunk+1,chunk+2])
            self.assertEqual(len(seen),2)
            self.assertEqual(result['described'],chunk+2)

    def test_page_concept_refs_are_local_and_source_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            orch=self.stores(Path(directory)/'oma.db');ns='u1__fixture'
            for cid,namespace in [('linked',ns),('prerequisite',ns),('unrelated',ns),('foreign','u2__fixture')]:
                orch.concept._insert(MemoryItem(id=cid,namespace=namespace,store='concept',content=cid,
                    store_specific={'name':cid,'prerequisite_concept_ids':['prerequisite'] if cid=='linked' else []}))
            for sid,entity in [('doc_a','linked'),('doc_b','unrelated')]:
                orch.content._insert(MemoryItem(id=sid,namespace=ns,store='content',content='Text',source_doc_id=sid,entities=[entity],store_specific={'page_number':1}))
            with patch('provider_capacity.call',side_effect=AssertionError('read path must not call a model')):
                refs=p.section_concept_refs(orch,ns,{'source_refs':[{'source_id':'a','pages':[1]}]})
            self.assertEqual({r['concept_id'] for r in refs},{'linked','prerequisite'})

    def test_completed_concept_refinement_is_not_repeated_for_same_inputs(self):
        import oma_provider
        from unittest.mock import Mock
        pipeline=Mock()
        pipeline.collect_concept_mentions.return_value={'topic':{'doc_a'}}
        pipeline.run_folder_concept_pass.return_value={'concepts':1,'concepts_new':1,'concepts_merged':0}
        with tempfile.TemporaryDirectory() as directory,patch.object(oma_provider,'OMA_DB_PATH',Path(directory)/'oma.db'),patch.object(oma_provider,'is_oma_enabled',return_value=True),patch.object(oma_provider,'_content_ingest_pipeline',return_value=pipeline),patch.object(oma_provider,'_background_concept_refine_started',set()),patch.object(oma_provider,'_tier_b_pending',set()),patch.object(oma_provider.threading,'Thread',side_effect=lambda target,**kwargs:SimpleNamespace(start=target)):
            oma_provider.kickoff_background_concept_refinement_async(1,'Physics')
            oma_provider.kickoff_background_concept_refinement_async(1,'Physics')
            self.assertEqual(pipeline.run_folder_concept_pass.call_count,1)
            pipeline.collect_concept_mentions.return_value={'topic':{'doc_a'},'new topic':{'doc_b'}}
            oma_provider.kickoff_background_concept_refinement_async(1,'Physics')
            self.assertEqual(pipeline.run_folder_concept_pass.call_count,2)

    def test_memory_jobs_are_claimed_independently_of_source_jobs(self):
        with SessionLocal() as db:
            learning_jobs.enqueue(db,1,'Physics',0)
            db.add(LearningJob(id='source-test',payload_json=json.dumps({'kind':'source_ingest','source_id':'source-test'}),status='queued',attempts=0,available_at=0,lease_until=0));db.commit()
        source=learning_jobs.claim('source');completion=learning_jobs.claim('completion')
        self.assertEqual(source[0],'source-test');self.assertNotEqual(completion[0],'source-test')
    def test_progressive_outline_does_not_wait_for_full_index(self):
        import lesson,oma_provider
        with SessionLocal() as db:
            db.add(FolderSource(user_id=1,folder_name='Physics',source_id='srcA',title='A',filename='a.pdf',source_type='pdf',page_count=8,raw_text='course',file_path='a.pdf',oma_ingest_status='INGESTING'));db.commit()
        with patch.object(oma_provider,'is_oma_enabled',return_value=True),patch.object(p,'manifest',return_value=self.manifest(8)),patch.object(lesson,'_call_llm_for_outline',return_value=[{'title':'First','source_units':['srcA:1-8'],'estimated_minutes':20}]),patch.object(oma_provider,'ensure_oma_ready_for_outline',side_effect=AssertionError('must not wait for full index')):
            result=lesson.generate_outline(1,'Physics')
        self.assertNotIn('error',result)
        self.assertEqual(result['outline_source'],'oma_progressive')
        self.assertEqual(result['sections'][0]['source_refs'][0]['pages'],list(range(1,9)))

if __name__=='__main__': unittest.main(verbosity=2)
