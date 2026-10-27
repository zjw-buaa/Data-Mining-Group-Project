"""Nine data-backed diagnostic figures; no synthetic agent series."""
from __future__ import annotations
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.plot_style import apply_publication_style, export_figure, COLOR_SEQUENCE, choose_font
from utils.setup_style import setup_style
from .quality import parse, FEATURES

NAMES = {'minimum': '最低必要处理', 'fixed': '固定规则', 'agent': '智能体'}


def make_figures(raw, groups, quality, output, emit):
    apply_publication_style(language='zh', width='double')
    setup_style(journal='general', lang='zh')
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'legend.fontsize': 9, 'svg.fonttype': 'none',
                         'font.sans-serif': [choose_font('zh'), 'DejaVu Sans']})
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    contracts = []
    p = parse(raw)
    colors = {name: COLOR_SEQUENCE[i] for i, name in enumerate(groups)}
    size = (7.2, 4.5)

    def new():
        return plt.subplots(figsize=size, layout='constrained')

    def finish(fig, stem, claim, source, caption, chart_type):
        # Vendored skill helper enforces layout and design gates before saving.
        paths = export_figure(fig, folder/stem, dpi=300, grayscale_preview=True,
                              strict_layout=True, strict_design=True)
        contracts.append({'id': stem, 'question': 'q1' if '_q1_' in stem else 'q2',
                          'core_claim': claim, 'source': source, 'caption': caption,
                          'layout': 'single quantitative panel', 'primary_panel': 'only panel',
                          'backend': 'Python/matplotlib', 'chart_type': chart_type,
                          'size_inches': size, 'dpi': 300, 'formats': ['svg', 'png'], 'paths': {k: Path(v).name for k,v in paths.items()}})
        emit('figure_exported', figure=stem, layout_and_design_gate='passed', size_inches=size, dpi=300)
        plt.close(fig)

    fig, ax = new()
    issues = {'评分缺失': quality['missing_by_column']['Rating'],
              '完全重复多余记录': quality['exact_duplicate_extra'],
              '核心异常记录': quality['quarantined_rows'],
              '大小随设备变化': quality['size_states'].get('varies', 0)}
    ax.barh(list(issues), list(issues.values()), color=COLOR_SEQUENCE[0])
    ax.set(xlabel='原始记录数（条）', ylabel='问题或特殊状态')
    ax.set_xlim(0, max(issues.values())*1.15)
    for i, v in enumerate(issues.values()):
        ax.text(v+max(issues.values())*.015, i, str(v), va='center', fontsize=9)
    finish(fig, 'raw_q1_quality', '原始数据同时存在缺失、重复、错位和特殊格式状态。',
           'quality_before.json', f'原始n={len(raw)}；各项可重叠，不应相加；大小随设备变化是特殊状态，不直接等于错误。计数，无误差棒。', 'horizontal_bar')

    fig, ax = new()
    ratings = p.loc[~p.quarantined, 'rating_observed'].dropna()
    ax.hist(ratings, bins=np.linspace(0,5,26), color=COLOR_SEQUENCE[0], edgecolor='white', linewidth=.4)
    ax.set(xlabel='有效观测评分（分）', ylabel='记录数（条）', xlim=(0,5))
    finish(fig, 'raw_q2_rating', '原始有效评分并非均匀分布。', 'data/raw/googleplaystore.csv',
           f'n={len(ratings)}条有效非空评分；排除核心异常；重复保留；未插补；直方图宽0.2分。', 'histogram')

    fig, ax = new()
    valid = p.loc[~p.quarantined]
    ax.scatter(np.log10(valid.reviews_value+1), np.log10(valid.installs_value+1),
               s=5, alpha=.14, color=COLOR_SEQUENCE[0], edgecolors='none')
    ax.set(xlabel='log10(评论数 + 1)', ylabel='log10(安装量下界 + 1)')
    finish(fig, 'raw_q2_reviews_installs', '评论数与安装量存在关联且跨越多个数量级。', 'data/raw/googleplaystore.csv',
           f'n={len(valid)}条核心有效记录；重复保留；每点一条记录，无回归或显著性检验；安装量为下界，不作因果解释。', 'scatter')

    fig, ax = new()
    stages = ['原始输入', '清洗后保留', '进入聚类']
    y = np.arange(3)
    height = .65/len(groups)
    for i, (name, g) in enumerate(groups.items()):
        s = g['summary']
        counts = [s['raw_rows'], s['cleaned_rows'], s['cluster_rows']]
        ax.barh(y + (i-(len(groups)-1)/2)*height, counts, height=height,
                color=colors[name], label=NAMES[name])
    ax.set_yticks(y, stages)
    ax.set(xlabel='记录数（条）', ylabel='流程阶段', xlim=(0, len(raw)*1.05))
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncols=2)
    finish(fig, 'process_q1_retention', '不同策略改变参与聚类的样本构成。', 'comparison.csv',
           '清洗保留数含暂不适合聚类的记录；进入聚类数才是聚类保留率分子。计数，非统计均值。', 'grouped_horizontal_bar')

    fig, ax = new()
    cats = ['评分代理填补', '大小代理填补']
    x = np.arange(2)
    width = .65/len(groups)
    for i, (name, g) in enumerate(groups.items()):
        s = g['summary']
        ax.bar(x+(i-(len(groups)-1)/2)*width, [s['rating_imputed'], s['size_imputed']],
               width=width, color=colors[name], label=NAMES[name])
    ax.set_xticks(x, cats)
    ax.set(xlabel='建模代理操作', ylabel='受影响记录数（条）')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncols=2)
    finish(fig, 'process_q1_imputation', '填补提高可计算性，但不恢复真实观测值。', 'comparison.csv',
           '统计填补操作数；同一记录可能两项均填补；缺失和varies标记保留。计数，无误差棒。', 'grouped_bar')

    fig, ax = new()
    markers = ['o', 's', '^']
    for i, name in enumerate(groups):
        ks = pd.read_csv(output/name/'k_candidates.csv')
        ax.scatter(ks.k + (i-(len(groups)-1)/2)*.04, ks.silhouette, marker=markers[i],
                   color=colors[name], label=NAMES[name], s=38)
    ax.set_xticks([2,3,4,5,6])
    ax.set(xlabel='候选簇数 k（横向微移仅防遮挡）', ylabel='轮廓系数（固定随机抽样）')
    ax.legend(loc='best')
    finish(fig, 'process_q2_k_selection', '按统一候选范围检查簇数敏感性。', '*/k_candidates.csv',
           '每组k=2至6；种子42，n_init=10；轮廓系数至多1500条固定样本；各组原生样本不同，跨组数值仅作描述。', 'dot')

    fig, ax = new()
    data = np.array([[g['summary'][f] for f in ['remaining_exact_duplicates', 'remaining_invalid_core_rows',
                                               'observed_rating_missing', 'observed_size_unavailable']] for g in groups.values()])
    im = ax.pcolormesh(np.arange(data.shape[1]), np.arange(data.shape[0]), data,
                       shading='nearest', cmap='cividis', vmin=0)
    ax.invert_yaxis()
    for (i,j), v in np.ndenumerate(data):
        ax.text(j, i, str(v), ha='center', va='center', color='white' if v < data.max()*.5 else 'black', fontsize=10)
    ax.set_xticks(range(4), ['完全重复多余记录', '核心非法记录', '原评分未观测', '原大小未观测'])
    ax.tick_params(axis='x', labelsize=8)
    ax.set_yticks(range(len(groups)), [NAMES[n] for n in groups])
    ax.set(xlabel='清洗后保留的错误或未知状态', ylabel='处理组')
    cbar = fig.colorbar(im, ax=ax, label='记录数（条）', shrink=.75)
    cbar.solids.set_rasterized(False)
    finish(fig, 'result_q1_remaining', '错误记录与原始未知状态必须分别报告。', '*/quality_after.json',
           '分母为各组清洗后记录；“原评分/大小未观测”即使填补也仍计数，不代表聚类矩阵有NaN。热图为计数，无显著性推断。', 'heatmap')

    fixed = groups['fixed']['native']
    fig, ax = new()
    centers = fixed['model'].cluster_centers_
    bound = float(np.max(np.abs(centers)))
    im = ax.pcolormesh(np.arange(centers.shape[1]), np.arange(centers.shape[0]), centers,
                       shading='nearest', cmap='RdBu_r', vmin=-bound, vmax=bound)
    ax.invert_yaxis()
    for (i, j), v in np.ndenumerate(centers):
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=10,
                color='white' if abs(v) > bound*.55 else 'black')
    ax.set_xticks(range(5), ['评分', '评论数', '大小', '安装量下界', '价格'])
    counts = fixed['assignments'].cluster.value_counts()
    ax.set_yticks(range(len(centers)), [f'簇{i} (n={counts[i]})' for i in range(len(centers))])
    ax.set(xlabel='共同变换后的特征', ylabel='固定规则分群')
    cbar = fig.colorbar(im, ax=ax, label='簇中心（标准化单位）', shrink=.75)
    cbar.solids.set_rasterized(False)
    finish(fig, 'result_q2_centers', '固定规则组的簇在规模、价格和评分等特征上不同。', 'fixed/centers_standardized.csv',
           'KMeans簇中心为变换空间均值；除评分外先log1p；包含建模代理值，非原单位均值；标签编号无业务排序。', 'heatmap')

    fig, ax = new()
    x = fixed['x']
    embedding_model = PCA(n_components=2, svd_solver='full').fit(x)
    points = embedding_model.transform(x)
    center_points = embedding_model.transform(centers)
    labels = fixed['model'].labels_
    sample = np.sort(np.random.default_rng(42).choice(len(x), min(2500, len(x)), replace=False))
    pd.DataFrame({'source_row_id': fixed['assignments'].iloc[sample].source_row_id.to_numpy()}).to_csv(output/'pca_display_sample_ids.csv', index=False)
    mark = ['o', 's', '^', 'v', 'D', 'P']
    for i in range(len(centers)):
        selected = sample[labels[sample] == i]
        ax.scatter(points[selected,0], points[selected,1], s=9, alpha=.28,
                   color=COLOR_SEQUENCE[i], marker=mark[i], edgecolors='none')
        ax.annotate(f'簇{i}', center_points[i], fontsize=10, weight='bold')
    ev = embedding_model.explained_variance_ratio_*100
    ax.set(xlabel=f'主成分1（解释方差{ev[0]:.1f}%）', ylabel=f'主成分2（解释方差{ev[1]:.1f}%）')
    finish(fig, 'result_q2_pca', '二维投影用于查看固定规则分群的重叠和位置。', 'fixed/clusters.csv; shared_scaler.json',
           f'PCA拟合全部固定规则组记录，仅绘制种子42抽取的{len(sample)}点；颜色和形状表示类别，文字标注簇中心。投影不用于重新聚类，不证明真实类别或因果。', 'scatter')

    (folder/'图表契约.json').write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding='utf-8')
    captions = ['# 图表说明', '', 'q1为四类质量问题处理，q2为应用分群与比较。全部图来自本次实际输出，不包含未运行的智能体序列。', '']
    html = ['<!doctype html><meta charset="utf-8"><title>Google Play 首轮实验图表</title>',
            '<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto;color:#222}img{width:100%;max-width:864px}section{margin:36px 0}p{line-height:1.6}</style>',
            '<h1>Google Play 首轮实验图表</h1><p>来源为本目录对应运行结果。智能体未运行时不会显示其结果。</p>']
    for c in contracts:
        captions += [f'## {c["id"]}', '', c['core_claim'], '', c['caption'], '', f'数据位置：{c["source"]}', '']
        html += [f'<section><h2>{c["core_claim"]}</h2><img src="{c["id"]}.png"><p>{c["caption"]}</p></section>']
    (folder/'图表说明.md').write_text('\n'.join(captions), encoding='utf-8')
    (folder/'图表面板.html').write_text('\n'.join(html), encoding='utf-8')
