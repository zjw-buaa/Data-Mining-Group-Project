"""Replay a validated real-model trace offline; never makes another API call.

The original live run remains the evidence for model usage and billing.
This command reproduces deterministic tools and clustering from that decision.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd
from threadpoolctl import threadpool_limits
from src.quality import MINIMUM, FIXED, read_raw, parse, report, validate_policy, process
from src.clustering import fit_reference, native, common_compare

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDED = ROOT/'results/final'
GROUPS = ('minimum', 'fixed', 'agent')
FINAL_CSVS = tuple(
    f'{group}/{name}.csv'
    for group in GROUPS
    for name in ('cleaned', 'quarantine', 'changes', 'clusters',
                 'cluster_profiles', 'representative_apps', 'k_candidates')
) + ('comparison.csv', 'common_metrics.csv', 'common_group_ari.csv',
     'common_seed_stability.csv', 'fixed/centers_standardized.csv')
FINAL_JSONS = tuple(
    f'{group}/{name}.json'
    for group in GROUPS for name in ('quality_after', 'selected_model')
) + ('quality_before.json', 'shared_scaler.json')
TIMING = {'seconds', 'processing_seconds', 'clustering_seconds'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(recorded):
    status = json.loads((recorded/'agent_status.json').read_text(encoding='utf-8'))
    manifest = json.loads((recorded/'run_manifest.json').read_text(encoding='utf-8'))
    if status.get('status') != 'completed' or status.get('real_model_called') is not True:
        raise ValueError('回放来源必须是已完成的真实模型运行')
    raw_path = ROOT/'data/raw/googleplaystore.csv'
    if digest(raw_path) != manifest['source_sha256']:
        raise ValueError('原始输入哈希与真实运行不一致')
    for name, expected in manifest['source_code_sha256'].items():
        if digest(ROOT/Path(name.replace('\\', '/'))) != expected:
            raise ValueError('真实运行后核心源码发生变化：'+name)
    events = [json.loads(x) for x in (recorded/'logs/pipeline.jsonl').read_text(encoding='utf-8').splitlines()]
    requests = [e for e in events if e['event'] == 'llm_request']
    responses = [e for e in events if e['event'] == 'llm_response']
    if len(requests) != status['calls'] or not responses:
        raise ValueError('调用次数或响应证据缺失')
    accepted = None
    for response in responses:
        data = response['response']
        try:
            candidate = json.loads(data['choices'][0]['message']['content'])
            if candidate == status['validated_plan']:
                accepted = response
        except (ValueError, KeyError, IndexError):
            pass
    if accepted is None:
        raise ValueError('未找到与已执行计划一致的原始模型响应')
    matching = [r for r in requests if r['attempt'] == accepted['attempt']]
    if len(matching) != 1 or not matching[0]['endpoint'].startswith('https://api.deepseek.com/'):
        raise ValueError('缺少对应DeepSeek请求')
    payload = json.loads(matching[0]['body']['messages'][1]['content'])
    if payload['quality'] != json.loads((recorded/'quality_before.json').read_text(encoding='utf-8')):
        raise ValueError('模型所读质量统计与原始核查不一致')
    validate_policy(status['validated_plan']['policy'])
    return status, manifest, events, accepted


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def verify_final_results(generated, recorded):
    """Check an explicit required set; a missing result must never be skipped."""
    for relative in (*FINAL_CSVS, *FINAL_JSONS):
        for folder in (generated, recorded):
            if not (folder/relative).is_file():
                raise FileNotFoundError(f'缺少最终结果：{folder/relative}')
    for relative in FINAL_CSVS:
        left, right = [pd.read_csv(folder/relative, dtype=str, keep_default_na=False)
                       for folder in (generated, recorded)]
        pd.testing.assert_frame_equal(
            left.drop(columns=list(TIMING), errors='ignore'),
            right.drop(columns=list(TIMING), errors='ignore'),
            obj=relative,
        )
    for relative in FINAL_JSONS:
        left, right = [json.loads((folder/relative).read_text(encoding='utf-8'))
                       for folder in (generated, recorded)]
        if without_timing(left) != without_timing(right):
            raise AssertionError('最终统计与真实运行不一致：'+relative)
    return {'csv': list(FINAL_CSVS), 'json': list(FINAL_JSONS)}


def recalculate(output, status, emit):
    """Recompute from raw observations using the recorded model decision."""
    def write_json(relative, value):
        (output/relative).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    raw = read_raw(ROOT/'data/raw/googleplaystore.csv')
    write_json('quality_before.json', report(raw))
    scaler, scaler_meta = fit_reference(parse(raw))
    write_json('shared_scaler.json', scaler_meta)
    groups, metric_rows = {}, []
    policies = {'minimum': MINIMUM, 'fixed': FIXED, 'agent': status['validated_plan']['policy']}
    with threadpool_limits(limits=1):
        for name, policy in policies.items():
            folder = output/name
            folder.mkdir()
            group = process(raw, policy, name, emit)
            for field in ('cleaned', 'quarantine', 'changes'):
                group[field].to_csv(folder/(field+'.csv'), index=False, encoding='utf-8-sig')
            group['native'] = native(group['cluster'], scaler, folder, name, emit)
            write_json(f'{name}/quality_after.json', group['summary'])
            write_json(f'{name}/selected_model.json', group['native']['selected'])
            summary = {k: v for k, v in group['summary'].items() if not isinstance(v, dict)}
            metric_rows.append({**summary, **group['native']['selected'],
                                'clustering_seconds': group['native']['clustering_seconds'],
                                'status': 'completed',
                                'model_calls': status['calls'] if name == 'agent' else 0,
                                'model_cost_usd': status.get('cost_usd') if name == 'agent' else 0})
            groups[name] = group
        common_compare(groups, scaler, output, emit)
    pd.DataFrame(metric_rows).to_csv(output/'comparison.csv', index=False, encoding='utf-8-sig')


def main(argv=None):
    p = argparse.ArgumentParser(description='离线复现真实DeepSeek计划；不调用API、不产生模型费用')
    p.add_argument('--recorded-run', type=Path, default=DEFAULT_RECORDED)
    p.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    recorded = args.recorded_run.resolve()
    status, manifest, _, accepted = load_trace(recorded)
    output = (args.out or ROOT/'results/replay'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('请使用新的空输出目录')
    output.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir()
    # Explicitly prevent network connections during replay, even if the environment has credentials.
    def no_network(*a, **kw):
        raise RuntimeError('离线回放禁止网络连接')
    def emit(event, **payload):
        with (output/'logs/replay.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event': event, **payload}, ensure_ascii=False, allow_nan=False)+'\n')
    # Publish final numeric outputs only after every required comparison passes.
    with tempfile.TemporaryDirectory(prefix='work_', dir=output) as temp:
        generated = Path(temp)
        with patch('socket.create_connection', no_network), patch('socket.socket.connect', no_network):
            recalculate(generated, status, emit)
        checks = verify_final_results(generated, recorded)
        for relative in (*FINAL_CSVS, *FINAL_JSONS):
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated/relative, target)
    receipt = {'mode': 'offline_replay_of_real_model_trace', 'new_api_calls': 0,
               'original_real_model_calls': status['calls'], 'original_response_id': accepted['response'].get('id'),
               'recorded_run': str(recorded), 'source_sha256': manifest['source_sha256'],
               'source_manifest_sha256': digest(recorded/'run_manifest.json'),
               'source_trace_sha256': digest(recorded/'logs/pipeline.jsonl'),
               'replay_script_sha256': digest(Path(__file__).resolve()),
               'policy': status['validated_plan']['policy'], 'csvs_equal_except_timing': checks['csv'],
               'jsons_equal_except_timing': checks['json'],
               'reproduce_command': 'python run.py --replay',
               'note': '本次只回放真实记录的已校验计划，不重新调用模型；不声称新模型实验或重复调用结果确定。'}
    (output/'replay_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'verified', 'new_api_calls': 0, 'csvs': len(checks['csv']),
                      'jsons': len(checks['json']), 'output': str(output)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
