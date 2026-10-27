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
import socket

import pandas as pd
from threadpoolctl import threadpool_limits
from src.quality import MINIMUM, FIXED, read_raw, parse, report, validate_policy, process
from src.clustering import fit_reference, native, common_compare
from src.plots import make_figures

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDED = ROOT/'results/run_/20261026_225712_978517'


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


def main(argv=None):
    p = argparse.ArgumentParser(description='离线复现真实DeepSeek计划；不调用API、不产生模型费用')
    p.add_argument('--recorded-run', type=Path, default=DEFAULT_RECORDED)
    p.add_argument('--out', type=Path)
    p.add_argument('--no-figures', action='store_true')
    args = p.parse_args(argv)
    recorded = args.recorded_run.resolve()
    status, manifest, original_events, accepted = load_trace(recorded)
    output = (args.out or ROOT/'results'/('replay_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('请使用新的空输出目录')
    output.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir()
    # Explicitly prevent network connections during replay, even if the environment has credentials.
    def no_network(*a, **kw):
        raise RuntimeError('离线回放禁止网络连接')
    socket.create_connection = no_network
    socket.socket.connect = no_network
    def emit(event, **payload):
        with (output/'logs/replay.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event': event, **payload}, ensure_ascii=False, allow_nan=False)+'\n')
    raw = read_raw(ROOT/'data/raw/googleplaystore.csv')
    quality = report(raw)
    original_quality = json.loads((recorded/'quality_before.json').read_text(encoding='utf-8'))
    if quality != original_quality:
        raise ValueError('重新检查数据得到的质量统计与真实模型输入不一致')
    scaler, scaler_meta = fit_reference(parse(raw))
    groups = {}
    policies = {'minimum': MINIMUM, 'fixed': FIXED, 'agent': status['validated_plan']['policy']}
    with threadpool_limits(limits=1):
        for name, policy in policies.items():
            folder = output/name
            folder.mkdir()
            group = process(raw, policy, name, emit)
            for field in ['cleaned', 'quarantine', 'changes']:
                group[field].to_csv(folder/(field+'.csv'), index=False, encoding='utf-8-sig')
            group['native'] = native(group['cluster'], scaler, folder, name, emit)
            groups[name] = group
        common_compare(groups, scaler, output, emit)
        if not args.no_figures:
            make_figures(raw, groups, quality, output, emit)
    checks = []
    for path in sorted(output.rglob('*.csv')):
        relative = path.relative_to(output)
        original = recorded/relative
        if not original.is_file():
            raise ValueError('真实运行中未找到对照文件：'+str(relative))
        left, right = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in [path, original]]
        timing = [c for c in left if c in {'seconds', 'processing_seconds', 'clustering_seconds'}]
        pd.testing.assert_frame_equal(left.drop(columns=timing), right.drop(columns=timing))
        checks.append(relative.as_posix())
    image_checks = []
    if not args.no_figures:
        from PIL import Image, ImageChops
        for path in sorted((output/'figures').rglob('*.png')):
            relative = path.relative_to(output)
            with Image.open(path) as a, Image.open(recorded/relative) as b:
                if a.size != b.size or ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is not None:
                    raise ValueError('PNG像素与真实运行不一致：'+str(relative))
            image_checks.append(relative.as_posix())
    receipt = {'mode': 'offline_replay_of_real_model_trace', 'new_api_calls': 0,
               'original_real_model_calls': status['calls'], 'original_response_id': accepted['response'].get('id'),
               'recorded_run': str(recorded.relative_to(ROOT)), 'source_sha256': manifest['source_sha256'],
               'source_manifest_sha256': digest(recorded/'run_manifest.json'),
               'source_trace_sha256': digest(recorded/'logs/pipeline.jsonl'),
               'replay_script_sha256': digest(Path(__file__).resolve()),
               'policy': status['validated_plan']['policy'], 'csvs_equal_except_timing': checks,
               'pngs_pixel_identical': image_checks,
               'reproduce_command': 'python replay_recorded_run.py',
               'note': '本次只回放真实记录的已校验计划，不重新调用模型；不声称新模型实验或重复调用结果确定。'}
    (output/'replay_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'verified', 'new_api_calls': 0, 'csvs': len(checks),
                      'pngs': len(image_checks), 'output': str(output)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
