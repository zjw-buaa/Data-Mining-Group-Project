import copy,io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from extensions.reason_guard import audit_plan
from extensions.guarded_agent import get_plan
from src.quality import read_raw
ROOT=Path(__file__).resolve().parents[1]

def plan(size='category_median',reason='采用同类别中位数填补未知大小，保留原始状态。'):
    return {'policy':{'deduplicate':True,'rating':'drop','size':size},'reasons':{'deduplicate':'移除完全重复的多余记录。','rating':'缺失评分不填补，排除聚类并保留档案。','size':reason}}

def response(p):
    return io.StringIO(json.dumps({'id':'mock-response-unit-test-only','model':'test-model', 'usage':{'prompt_tokens':10,'completion_tokens':5},'choices':[{'message':{'content':json.dumps(p,ensure_ascii=False)}}]}))

class ReasonGuardTests(unittest.TestCase):
    def test_actual_four_saved_responses(self):
        for path,allowed in [('results/final',True),('results/stability/runs/02',True),('results/stability/runs/03',True),('results/stability/runs/04',False)]:
            p=json.loads((ROOT/path/'agent_status.json').read_text(encoding='utf-8'))['validated_plan']
            with self.subTest(path=path):
                report=audit_plan(p)
                self.assertEqual(report['allow_execution'],allowed)
                if not allowed:self.assertTrue(any(x['field']=='size' and x['code']=='policy_reason_conflict' for x in report['issues']))
    def test_reject_impute_vs_no_imputation(self):
        self.assertFalse(audit_plan(plan(reason='不适合用中位数填充，因此按类别取中位数。'))['allow_execution'])
    def test_plain_no_fill_cannot_be_followed_by_imputation(self):
        self.assertFalse(audit_plan(plan(reason='未知大小不填补，但采用类别中位数填补。'))['allow_execution'])
    def test_global_rejection_and_category_choice_are_distinct(self):
        self.assertTrue(audit_plan(plan(reason='不使用全局中位数填充，改用类别中位数填补。'))['allow_execution'])
    def test_category_fallback_does_not_conflict(self):
        self.assertTrue(audit_plan(plan(reason='使用类别中位数填补，不足20条时回退全局中位数。'))['allow_execution'])
    def test_drop_but_reason_imputes_is_rejected(self):
        self.assertFalse(audit_plan(plan('drop','采用类别中位数填补所有缺失大小。'))['allow_execution'])
    def test_no_dedupe_but_reason_removes_duplicates(self):
        p=plan();p['policy']['deduplicate']=False
        self.assertFalse(audit_plan(p)['allow_execution'])
    def test_dedupe_but_reason_keeps_duplicates(self):
        p=plan();p['reasons']['deduplicate']='保留所有重复记录，不去重。'
        self.assertFalse(audit_plan(p)['allow_execution'])
    def test_unknown_or_vague_reason_requires_clarification(self):
        self.assertFalse(audit_plan(plan(reason='这种办法比较合理，能够提升数据质量。'))['allow_execution'])
    def test_english_contradiction(self):
        self.assertFalse(audit_plan(plan(reason='Median imputation is not suitable for median imputation; use category median to impute.'))['allow_execution'])
    def test_global_policy_cannot_claim_category_action(self):
        self.assertFalse(audit_plan(plan('global_median','采用类别中位数填补缺失大小。'))['allow_execution'])
    def test_wrong_reason_type_is_structural_error(self):
        p=plan();p['reasons']=[]
        with self.assertRaises(ValueError):audit_plan(p)

class PlannerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.raw=read_raw(ROOT/'data/raw/googleplaystore.csv')
    def call(self,values):
        events=[]
        def emit(event,**payload):events.append((event,copy.deepcopy(payload)))
        env={'LLM_API_KEY':'mock-test-key','LLM_BASE_URL':'https://api.deepseek.com','LLM_MODEL':'test-model'}
        with patch.dict(os.environ,env,clear=True),patch('urllib.request.urlopen',side_effect=[response(p) for p in values]) as api:
            policy,status=get_plan({},self.raw,emit)
        return policy,status,events,api
    def test_conflict_then_corrected_retries_with_full_conversation(self):
        bad=plan(reason='不适合用中位数填充，因此按类别取中位数。')
        policy,status,events,api=self.call([bad,plan()])
        self.assertEqual(api.call_count,2);self.assertEqual(status['calls'],2)
        self.assertEqual(status['prompt_tokens'],20);self.assertEqual(status['completion_tokens'],10)
        self.assertEqual(policy,plan()['policy']);self.assertEqual(status['status'],'completed')
        requests=[d for e,d in events if e=='llm_request']
        self.assertEqual([m['role'] for m in requests[1]['body']['messages']],['system','user','assistant','user'])
        self.assertIn('理由与策略',requests[1]['body']['messages'][-1]['content'])
        self.assertEqual(requests[0]['body']['messages'][-1]['role'],'user')
        self.assertTrue(status['enhancement_source_sha256'])
    def test_two_conflicts_stop_no_policy(self):
        bad=plan(reason='不适合用中位数填充，因此按类别取中位数。')
        policy,status,events,api=self.call([bad,bad])
        self.assertIsNone(policy);self.assertEqual(status['status'],'invalid_plan')
        self.assertEqual(api.call_count,2);self.assertFalse(status['reason_guard']['allow_execution'])
        self.assertEqual([d['action'] for e,d in events if e=='llm_reason_conflict'],['retry','stop'])
    def test_malformed_then_corrected_shares_same_budget(self):
        policy,status,events,api=self.call([{'policy':{}},plan()])
        self.assertEqual(status['calls'],2);self.assertEqual(status['status'],'completed')
    def test_single_consistent_plan_has_no_extra_request(self):
        policy,status,events,api=self.call([plan()])
        self.assertEqual(api.call_count,1);self.assertTrue(status['reason_guard']['allow_execution'])
    def test_pipeline_does_not_execute_agent_after_two_conflicts(self):
        from run import main
        root = ROOT/'.cache/reason_guard_pipeline_test'
        root.mkdir(parents=True,exist_ok=True)
        latest=ROOT/'results/latest_run.txt'
        before=latest.read_bytes() if latest.exists() else None
        bad=plan(reason='不适合用中位数填充，因此按类别取中位数。')
        env={'LLM_API_KEY':'mock-test-key','LLM_BASE_URL':'https://api.deepseek.com','LLM_MODEL':'test-model'}
        try:
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                out=Path(tmp)/'out'
                with patch.dict(os.environ,env,clear=True),patch('urllib.request.urlopen',side_effect=[response(bad),response(bad)]),patch('builtins.print'):
                    code=main(['--agent','required','--no-figures','--out',str(out)])
                self.assertEqual(code,2)
                self.assertFalse((out/'agent').exists())
                self.assertTrue((out/'fixed/cleaned.csv').exists())
                status=json.loads((out/'agent_status.json').read_text(encoding='utf-8'))
                self.assertEqual(status['status'],'invalid_plan')
        finally:
            if before is not None:latest.write_bytes(before)
            elif latest.exists():latest.unlink()

    def test_no_key_does_not_call_api(self):
        with patch.dict(os.environ,{},clear=True),patch('urllib.request.urlopen') as api:
            policy,status=get_plan({},self.raw,lambda *a,**k:None)
        api.assert_not_called();self.assertIsNone(policy);self.assertEqual(status['status'],'not_configured')
    def test_transport_error_is_not_retried_as_a_reason_conflict(self):
        import urllib.error
        with patch.dict(os.environ,{'LLM_API_KEY':'mock','LLM_BASE_URL':'https://api.deepseek.com','LLM_MODEL':'test'},clear=True),patch('urllib.request.urlopen',side_effect=urllib.error.URLError('test')) as api:
            policy,status=get_plan({},self.raw,lambda *a,**k:None)
        self.assertIsNone(policy);self.assertEqual(status['status'],'api_error');self.assertEqual(api.call_count,1)

if __name__=='__main__':unittest.main()