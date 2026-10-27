"""Shared transform and explicit native/common-population comparisons."""
from __future__ import annotations
import itertools
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from .quality import FEATURES

SEED = 42
SEEDS = [42, 52, 62]
K_CANDIDATES = [2, 3, 4, 5, 6]
COMMON_K = 4
SILHOUETTE_N = 1500


def values(frame):
    x = frame[FEATURES].to_numpy(float).copy()
    x[:, 1:] = np.log1p(x[:, 1:])
    return x


def fit_reference(raw_parsed):
    ref = raw_parsed.loc[~raw_parsed.quarantined & ~raw_parsed.exact_duplicate_extra].copy()
    ref['rating_value'] = ref.rating_observed
    ref['size_mb'] = ref.size_mb_observed
    ref = ref.dropna(subset=FEATURES)
    if len(ref) < 10:
        raise ValueError('有效完整参考记录不足10条，无法拟合公共缩放器')
    scaler = StandardScaler().fit(values(ref))
    meta = {'features': FEATURES, 'log1p_features': FEATURES[1:], 'reference_rows': len(ref),
            'reference_source_row_ids': ref.source_row_id.astype(int).tolist(),
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'description': 'Same complete, valid, exact-deduplicated raw reference for every group.'}
    return scaler, meta


def score(x, labels, sample_indices):
    n_labels = len(np.unique(labels))
    if not 1 < n_labels < len(x):
        raise ValueError('聚类退化，无法计算分群指标')
    sx, sl = x[sample_indices], labels[sample_indices]
    silhouette = float(silhouette_score(sx, sl)) if 1 < len(np.unique(sl)) < len(sl) else None
    return {'silhouette': silhouette, 'silhouette_sample_n': len(sample_indices),
            'davies_bouldin': float(davies_bouldin_score(x, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(x, labels))}


def fit_one(x, k, seed, sample):
    if len(np.unique(x, axis=0)) < k:
        raise ValueError('不同特征向量数小于簇数')
    started = time.perf_counter()
    model = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300, tol=1e-4).fit(x)
    if model.n_iter_ >= 300:
        raise RuntimeError('KMeans触及最大迭代次数，需检查收敛')
    return model, {**score(x, model.labels_, sample), 'k': k, 'seed': seed,
                   'inertia': float(model.inertia_), 'iterations': int(model.n_iter_),
                   'converged_before_max_iter': bool(model.n_iter_ < 300),
                   'seconds': time.perf_counter()-started}


def native(frame, scaler, out, group, emit):
    started = time.perf_counter()
    x = scaler.transform(values(frame))
    sample = np.sort(np.random.default_rng(SEED).choice(len(x), min(SILHOUETTE_N, len(x)), replace=False))
    pd.DataFrame({'source_row_id': frame.iloc[sample].source_row_id.to_numpy()}).to_csv(out/'silhouette_sample_ids.csv', index=False)
    runs = []
    models = {}
    for k in K_CANDIDATES:
        model, stats = fit_one(x, k, SEED, sample)
        runs.append(stats)
        models[k] = model
        emit('clustering_candidate', group=group, scope='native', **stats)
    selected = max(runs, key=lambda s: (-np.inf if s['silhouette'] is None else s['silhouette'], -s['k']))
    model = models[selected['k']]
    pd.DataFrame(runs).to_csv(out/'k_candidates.csv', index=False)
    assignments = frame[['source_row_id', 'App', 'Category', *FEATURES,
                         'rating_missing', 'size_state', 'rating_imputed', 'size_imputed']].copy()
    assignments['cluster'] = model.labels_
    assignments.to_csv(out/'clusters.csv', index=False, encoding='utf-8-sig')
    profiles = assignments.groupby('cluster')[FEATURES].median()
    profiles.insert(0, 'n', assignments.groupby('cluster').size())
    profiles['rating_imputed_rate'] = assignments.groupby('cluster').rating_imputed.mean()
    profiles['size_imputed_rate'] = assignments.groupby('cluster').size_imputed.mean()
    profiles.to_csv(out/'cluster_profiles.csv', encoding='utf-8-sig')
    categories = assignments.groupby(['cluster', 'Category']).size().rename('n').reset_index()
    categories.to_csv(out/'cluster_categories.csv', index=False, encoding='utf-8-sig')
    reps = []
    for k, center in enumerate(model.cluster_centers_):
        members = np.flatnonzero(model.labels_ == k)
        dist = np.linalg.norm(x[members]-center, axis=1)
        for pos in np.argsort(dist)[:3]:
            row = assignments.iloc[members[pos]]
            reps.append({'cluster': k, 'source_row_id': int(row.source_row_id), 'App': row.App,
                         'Category': row.Category, 'distance_to_center': float(dist[pos])})
    pd.DataFrame(reps).to_csv(out/'representative_apps.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(model.cluster_centers_, columns=FEATURES).to_csv(out/'centers_standardized.csv', index_label='cluster')
    return {'selected': selected, 'clustering_seconds': time.perf_counter()-started,
            'assignments': assignments, 'x': x, 'model': model}


def common_compare(groups, scaler, out, emit):
    ids = sorted(set.intersection(*(set(g['cluster'].source_row_id) for g in groups.values())))
    if len(ids) < COMMON_K+1:
        raise ValueError('共同样本不足以运行固定簇数比较')
    pd.DataFrame({'source_row_id': ids}).to_csv(out/'common_source_row_ids.csv', index=False)
    sample = np.sort(np.random.default_rng(SEED).choice(len(ids), min(SILHOUETTE_N, len(ids)), replace=False))
    pd.DataFrame({'source_row_id': np.array(ids)[sample]}).to_csv(out/'common_silhouette_sample_ids.csv', index=False)
    rows, labels, transformed = [], {}, {}
    stability = []
    for name, group in groups.items():
        frame = group['cluster'].set_index('source_row_id').loc[ids].reset_index()
        x = scaler.transform(values(frame))
        transformed[name] = x
        seed_labels = []
        for seed in SEEDS:
            model, stats = fit_one(x, COMMON_K, seed, sample)
            rows.append({'group': name, 'common_rows': len(ids), **stats})
            emit('clustering_common', group=name, **stats, common_rows=len(ids))
            seed_labels.append(model.labels_)
            if seed == SEED:
                labels[name] = model.labels_
                pd.DataFrame({'source_row_id': ids, 'cluster': model.labels_}).to_csv(out/f'common_clusters_{name}.csv', index=False)
        for (a, la), (b, lb) in itertools.combinations(zip(SEEDS, seed_labels), 2):
            stability.append({'group': name, 'seed_a': a, 'seed_b': b,
                              'adjusted_rand_index': float(adjusted_rand_score(la, lb))})
    pd.DataFrame(rows).to_csv(out/'common_metrics.csv', index=False)
    pd.DataFrame(stability).to_csv(out/'common_seed_stability.csv', index=False)
    pairs = []
    for a, b in itertools.combinations(labels, 2):
        pairs.append({'group_a': a, 'group_b': b, 'common_rows': len(ids), 'k': COMMON_K,
                      'adjusted_rand_index': float(adjusted_rand_score(labels[a], labels[b])),
                      'max_abs_transformed_feature_difference': float(np.max(np.abs(transformed[a]-transformed[b])))})
    pd.DataFrame(pairs).to_csv(out/'common_group_ari.csv', index=False)
    return rows, pairs
