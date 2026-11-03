"""Read-only application of new reason rules to historical real-model responses."""
import argparse,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
from extensions.reason_guard import audit_plan,VERSION,LIMITATION
from scripts.replay import load_trace
ROOT=Path(__file__).resolve().parents[1]
def main(argv=None):
    parser=argparse.ArgumentParser(description='只读审计四次已保存理由；不调用API，不修改历史记录')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    output=(args.out or ROOT/'results/reason_guard_audits'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('审计输出目录必须为空')
    results=[]
    for name,folder in [('main',ROOT/'results/final'),*[(i,ROOT/'results/stability/runs'/i) for i in ['02','03','04']]]:
        status,manifest,events,response=load_trace(folder)
        audit=audit_plan(status['validated_plan'])
        results.append({'run':name,'historical_status':status['status'],'original_response_id':response['response']['id'],
                        'source_status_sha256':hashlib.sha256((folder/'agent_status.json').read_bytes()).hexdigest(),
                        'source_log_sha256':hashlib.sha256((folder/'logs/pipeline.jsonl').read_bytes()).hexdigest(),
                        'audit':audit})
    report={'mode':'retrospective_read_only_rule_audit','version':VERSION,'new_api_calls':0,
            'historical_experiments_modified':False,'limitation':LIMITATION,'runs':results}
    output.mkdir(parents=True,exist_ok=True)
    (output/'reason_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'audit_completed','new_api_calls':0,'runs':len(results),
                      'needs_clarification':[x['run'] for x in results if not x['audit']['allow_execution']],
                      'output':str(output)},ensure_ascii=False))
    return 0