from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .quality import COLUMNS, FEATURES, MINIMUM, FIXED, read_raw, report, parse, process
from .clustering import native, fit_reference, common_compare, SEED, SEEDS, K_CANDIDATES, COMMON_K
from .agent import get_plan

EXPECTED_SHA = '3e438f48161961933d26e99a8d9fc8ed79edfaa9fb34f8838e1ab4ec7a9fab91'
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def review_sheet(raw, fixed, output):
    p = parse(raw)
    items = []
    selected = set()
    def take(kind, frame, count, note):
        for _, r in frame.iterrows():
            if int(r.source_row_id) in selected:
                continue
            items.append((kind, r, note))
            selected.add(int(r.source_row_id))
            if sum(k == kind for k, _, _ in items) >= count:
                break
    take('缺失', p.loc[p.rating_missing & p.reviews_value.gt(0)], 2, '评分缺失但评论数>0，不推断原因')
    take('缺失', p.loc[p.rating_missing & p.reviews_value.eq(0)], 3, '评分缺失，不以0替代评分')
    take('重复', p.loc[p.exact_duplicate_extra], 3, '原始13列完全重复；核对保留首条')
    take('异常或噪声', p.loc[p.quarantined], 1, '整行隔离；保留每个冲突字段和原值')
    take('异常或噪声', p.loc[~p.quarantined].sort_values('installs_value', ascending=False), 2, '合法高安装量，用于核查是否误删热门应用')
    take('异常或噪声', p.loc[~p.quarantined].sort_values('price_usd', ascending=False), 3, '合法高价格，用于核查是否仅因数值大误删')
    for mask, note in [(p.Size.str.endswith('k'), 'k转MB，除以1000'),
                       (p.Size.str.endswith('M'), 'M转MB，数值保持'),
                       (p.size_varies, '保留varies状态；建模代理不是真实大小')]:
        take('类型或格式', p.loc[mask], sum(k == '类型或格式' for k, _, _ in items)+1, note)
    cleaned = fixed['cleaned'].set_index('source_row_id')
    assignments = fixed['native']['assignments'].set_index('source_row_id')
    rows = []
    for kind, r, note in items:
        rid = int(r.source_row_id)
        retained = rid in cleaned.index
        c = cleaned.loc[rid] if retained else None
        rows.append({'问题类型': kind, 'source_row_id': rid, 'App': r.App,
                     '原始Rating': r.Rating, '原始Reviews': r.Reviews, '原始Size': r.Size,
                     '原始Installs': r.Installs, '原始Price': r.Price,
                     '固定规则处理': '保留' if retained else ('隔离' if r.quarantined else '完全重复移除'),
                     'rating_value': c.rating_value if retained else '',
                     'size_mb': c.size_mb if retained else '',
                     'size_state': c.size_state if retained else r.size_state,
                     'cluster': int(assignments.loc[rid, 'cluster']) if rid in assignments.index else '',
                     '机器处理说明': note, '人工是否合理': '', '人工疑问': '', '复核人': ''})
    pd.DataFrame(rows).to_csv(output/'人工抽查12项_待赵思嘉复核.csv', index=False, encoding='utf-8-sig')


def summary_text(output, quality, groups, agent, common, pairs):
    lines = ['# 10月26日首轮运行记录', '',
             '本页由真实运行自动生成；运行时间与模型状态见run_manifest.json和agent_status.json。', '',
             f'原始数据：{quality["row_count"]}行、13列；Rating缺失{quality["missing_by_column"]["Rating"]}行；完全重复多余记录{quality["exact_duplicate_extra"]}行；核心异常隔离{quality["quarantined_rows"]}行。', '',
             '| 组别 | 清洗保留数 | 聚类行数 | 聚类保留率 | 选定k | 轮廓系数（抽样） | DB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, g in groups.items():
        s, n = g['summary'], g['native']['selected']
        lines.append(f'| {name} | {s["cleaned_rows"]} | {s["cluster_rows"]} | {s["cluster_retention"]:.2%} | {n["k"]} | {n["silhouette"]:.4f} | {n["davies_bouldin"]:.4f} |')
    lines += ['', f'智能体组状态：`{agent["status"]}`；真实模型调用次数：{agent["calls"]}；费用：'+('未知/不适用' if agent.get('cost_usd') is None else str(agent['cost_usd'])+' USD')+'。']
    if agent['status'] != 'completed':
        lines += ['智能体组尚未完成真实运行，本轮只有两组基线结果，不能写成“三组实验已完成”。']
    lines += ['', '## 结果如何解释', '',
              '原生样本指标同时受到处理策略和保留样本构成影响，不能仅凭轮廓系数断言哪组更好。中位数是建模代理，不是恢复出的真实评分或大小；原始缺失及varies状态仍可追溯。']
    for p in pairs:
        lines += [f'共同记录{p["common_rows"]}条，固定k=4时，{p["group_a"]}与{p["group_b"]}的ARI={p["adjusted_rand_index"]:.6f}，最大特征差={p["max_abs_transformed_feature_difference"]:.6g}。']
    lines += ['若两组共同样本特征和分群相同，说明当前可观察差别主要来自额外保留记录和权重，不是对完整样本发现了更好的结构。共同样本都是交集，不能据此判断被填补记录的真实准确性。', '',
              '## 明天接手事项', '',
              '赵思嘉：填写人工抽查12项CSV中的人工判断和疑问；这张表尚未经人工验收。',
              '戴想然：接模型配置后真实运行智能体组，按抽查修正，重新生成三组比较和日志。',
              '周静雯：可先使用质量报告、两组结果及图；保留模型未配置和样本口径限制。',
              '闫灿宇：27日晚集中检查结果与后续调整。', '',
              '## 文件', '',
              '`quality_before.json`为原始核查；每组目录有cleaned.csv、quarantine.csv、changes.csv、clusters.csv、cluster_profiles.csv和代表应用；comparison.csv为主比较表，common_*.csv为共同记录比较。logs/pipeline.jsonl为真实工具事件。']
    (output/'首轮结果说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Google Play质量处理与分群完整流程')
    parser.add_argument('--input', type=Path, default=ROOT/'data/raw/googleplaystore.csv')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--agent', choices=['auto', 'off', 'required'], default='auto')
    parser.add_argument('--smoke', action='store_true', help='真实数据的首段和末段小样本；不产生正式图')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args(argv)
    source = args.input.resolve()
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError('输入SHA256与已核查主表不符；请先单独核对输入版本')
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = (args.out or ROOT/'results'/('smoke_' if args.smoke else 'run_') / timestamp).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录非空；请用新的--out目录，避免混入旧智能体结果')
    output.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    event_path = output/'logs/pipeline.jsonl'

    def emit(event, **payload):
        entry = {'at_utc': datetime.now(timezone.utc).isoformat(), 'event': event, **payload}
        with event_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+'\n')

    emit('run_started', input_sha256=sha, smoke=args.smoke, seed=SEED)
    raw = read_raw(source, 800 if args.smoke else None)
    quality = report(raw)
    write_json(output/'quality_before.json', quality)
    scaler, scaler_meta = fit_reference(parse(raw))
    write_json(output/'shared_scaler.json', scaler_meta)
    policies = {'minimum': MINIMUM, 'fixed': FIXED}
    if args.agent == 'off' or args.smoke:
        agent = {'status': 'disabled', 'real_model_called': False, 'calls': 0, 'cost_usd': None}
    else:
        plan, agent = get_plan(quality, raw, emit)
        if plan:
            policies['agent'] = plan
    write_json(output/'agent_status.json', agent)
    emit('agent_status', **agent)
    groups = {}
    metric_rows = []
    with threadpool_limits(limits=1):
        for name, policy in policies.items():
            print('Running', name, flush=True)
            folder = output/name
            folder.mkdir()
            g = process(raw, policy, name, emit)
            for key in ['cleaned', 'quarantine', 'changes']:
                g[key].to_csv(folder/(key+'.csv'), index=False, encoding='utf-8-sig')
            same_name = g['cleaned'].loc[g['cleaned'].App.duplicated(keep=False), ['source_row_id', *COLUMNS]]
            same_name.to_csv(folder/'same_name_review.csv', index=False, encoding='utf-8-sig')
            g['native'] = native(g['cluster'], scaler, folder, name, emit)
            write_json(folder/'quality_after.json', g['summary'])
            write_json(folder/'selected_model.json', g['native']['selected'])
            groups[name] = g
            s = {k: v for k, v in g['summary'].items() if not isinstance(v, dict)}
            metric_rows.append({**s, **g['native']['selected'], 'clustering_seconds': g['native']['clustering_seconds'],
                                'status': 'completed', 'model_calls': agent['calls'] if name == 'agent' else 0,
                                'model_cost_usd': agent.get('cost_usd') if name == 'agent' else 0})
        common, pairs = common_compare(groups, scaler, output, emit)
    if 'agent' not in groups:
        metric_rows.append({'group': 'agent', 'status': agent['status'], 'model_calls': agent['calls'], 'model_cost_usd': None})
    pd.DataFrame(metric_rows).to_csv(output/'comparison.csv', index=False, encoding='utf-8-sig')
    review_sheet(raw, groups['fixed'], output)
    summary_text(output, quality, groups, agent, common, pairs)
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'Pillow', 'threadpoolctl']}
    reproduce_args = ['python', 'run.py', '--agent', args.agent]
    if args.smoke:
        reproduce_args.append('--smoke')
    if args.no_figures:
        reproduce_args.append('--no-figures')
    if source != (ROOT/'data/raw/googleplaystore.csv').resolve():
        reproduce_args += ['--input', str(source)]
    reproduce_command = subprocess.list2cmdline(reproduce_args)
    manifest = {'timestamp_asia_shanghai': timestamp, 'source_sha256': sha, 'source_records': len(raw),
                'actual_invocation': subprocess.list2cmdline(['python', 'run.py', *(argv if argv is not None else sys.argv[1:])]),
                'reproduce_command': reproduce_command,
                'reproduce_output_note': '复现保留实际模式和模型选项，省略--out以自动新建目录；结果数值可核对，时间戳与耗时会变化。API环境配置须相同。',
                'smoke': args.smoke, 'parameters': {'seed': SEED, 'stability_seeds': SEEDS,
                'k_candidates': K_CANDIDATES, 'common_k': COMMON_K, 'silhouette_sample_max': 1500,
                'n_init': 10, 'max_iter': 300, 'tol': 1e-4, 'threads': 1},
                'dependencies': versions, 'agent_status': agent['status'], 'completed_groups': list(groups),
                'source_code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((ROOT/'src').glob('*.py'))}}
    write_json(output/'run_manifest.json', manifest)
    if not args.smoke and not args.no_figures:
        from .plots import make_figures
        make_figures(raw, groups, quality, output, emit)
    from utils.repro_manifest import build_manifest
    repro = build_manifest([source], SEED, {**manifest['parameters'], 'smoke': args.smoke, 'agent_mode': args.agent,
                                          'no_figures': args.no_figures}, reproduce_command, list(versions))
    repro['actual_invocation'] = manifest['actual_invocation']
    repro['reproduce_output_note'] = manifest['reproduce_output_note']
    write_json(output/'复现清单.json', repro)
    emit('run_completed', output=str(output), completed_groups=list(groups), agent_status=agent['status'])
    if not args.smoke:
        (ROOT/'results/latest_run.txt').write_text(str(output.relative_to(ROOT)), encoding='utf-8')
    print(json.dumps({'output': str(output), 'groups': list(groups), 'agent_status': agent['status']}, ensure_ascii=False))
    return 2 if args.agent == 'required' and agent['status'] != 'completed' else 0
