"""Deterministic quality checks; original values and stable record IDs survive."""
from __future__ import annotations

import re
import time
import numpy as np
import pandas as pd

COLUMNS = ['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
           'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver', 'Android Ver']
FEATURES = ['rating_value', 'reviews_value', 'size_mb', 'installs_value', 'price_usd']
EMPTY = {'', 'nan', 'null', 'none', 'na', 'n/a'}
MINIMUM = {'deduplicate': False, 'rating': 'drop', 'size': 'drop'}
FIXED = {'deduplicate': True, 'rating': 'global_median', 'size': 'global_median'}


def missing(s):
    return s.fillna('').astype(str).str.strip().str.lower().isin(EMPTY)


def read_raw(path, limit=None):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != COLUMNS:
        raise ValueError('输入字段与主表13列不一致')
    if limit:
        # Include the known malformed row and tail even in the smoke fixture.
        ids = sorted(set(range(min(limit, len(df)))) | set(range(max(0, len(df)-400), len(df))))
        df = df.iloc[ids].copy()
    df.insert(0, 'source_row_id', df.index.to_numpy() + 1)
    return df.reset_index(drop=True)


def parse(raw):
    d = raw.copy()
    text = {c: d[c].fillna('').str.strip() for c in COLUMNS}
    d['rating_missing'] = missing(d['Rating'])
    d['rating_observed'] = pd.to_numeric(text['Rating'].where(~d.rating_missing), errors='coerce')
    d['reviews_value'] = pd.to_numeric(text['Reviews'].where(text['Reviews'].str.fullmatch(r'\d+')), errors='coerce')
    installs = text['Installs'].str.fullmatch(r'(?:\d+|\d{1,3}(?:,\d{3})+)\+?')
    d['installs_value'] = pd.to_numeric(text['Installs'].str.replace(',', '', regex=False).str.removesuffix('+').where(installs), errors='coerce')
    prices = text['Price'].str.fullmatch(r'\$?\d+(?:\.\d+)?')
    d['price_usd'] = pd.to_numeric(text['Price'].str.removeprefix('$').where(prices), errors='coerce')
    size = text['Size'].str.extract(r'^(\d+(?:\.\d+)?)(M|k)$')
    d['size_mb_observed'] = pd.to_numeric(size[0], errors='coerce') * size[1].map({'M': 1., 'k': .001})
    d['size_varies'] = text['Size'].eq('Varies with device')
    d['size_missing'] = missing(d['Size'])
    d['size_state'] = np.select([d.size_varies, d.size_missing, d.size_mb_observed.notna()], ['varies', 'missing', 'observed'], default='invalid')
    d['last_updated_iso'] = pd.to_datetime(text['Last Updated'], format='%B %d, %Y', errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
    d['same_name_multiple_records'] = d['App'].duplicated(keep=False)
    d['exact_duplicate_extra'] = d.duplicated(COLUMNS, keep='first')
    d['type_price_conflict'] = ((text['Type'].eq('Free') & d.price_usd.gt(0)) |
                                (text['Type'].eq('Paid') & d.price_usd.eq(0)))
    checks = {
        'rating_out_of_range_or_invalid': ~d.rating_missing & (~d.rating_observed.between(0, 5)),
        'reviews_invalid': d.reviews_value.isna(),
        'installs_invalid': d.installs_value.isna(),
        'price_invalid': d.price_usd.isna(),
        'size_invalid': d.size_state.eq('invalid'),
        'category_invalid': ~text['Category'].str.fullmatch('[A-Z_]+'),
        'app_missing': missing(d['App']),
    }
    reasons = [[] for _ in range(len(d))]
    for name, mask in checks.items():
        for i in np.flatnonzero(mask.to_numpy()):
            reasons[i].append(name)
    d['invalid_reasons'] = [';'.join(r) for r in reasons]
    d['quarantined'] = d.invalid_reasons.ne('')
    return d


def report(raw):
    d = parse(raw)
    return {
        'row_count': len(d), 'column_count': len(COLUMNS),
        'missing_by_column': {c: int(missing(d[c]).sum()) for c in COLUMNS},
        'exact_duplicate_extra': int(d.exact_duplicate_extra.sum()),
        'same_name_rows': int(d.same_name_multiple_records.sum()),
        'quarantined_rows': int(d.quarantined.sum()),
        'invalid_rating_rows': int((~d.rating_missing & ~d.rating_observed.between(0, 5)).sum()),
        'size_states': {str(k): int(v) for k, v in d.size_state.value_counts().items()},
        'missing_rating_positive_reviews': int((d.rating_missing & d.reviews_value.gt(0)).sum()),
        'date_parse_failures': int(d.last_updated_iso.eq('').sum()),
        'type_price_conflicts': int(d.type_price_conflict.sum()),
        'format_workload': {
            'size_M': int(d.Size.str.endswith('M').sum()),
            'size_k': int(d.Size.str.endswith('k').sum()),
            'size_varies': int(d.size_varies.sum()),
            'installs_plus': int(d.Installs.str.endswith('+').sum()),
            'price_dollar': int(d.Price.str.startswith('$').sum()),
        },
        'invalid_examples': d.loc[d.quarantined, ['source_row_id', *COLUMNS, 'invalid_reasons']].head(5).to_dict('records'),
        'scope': 'Original listing records; app name is not a verified unique entity identifier.',
    }


def validate_policy(policy):
    if set(policy) != {'deduplicate', 'rating', 'size'}:
        raise ValueError('计划字段必须为deduplicate/rating/size')
    if type(policy['deduplicate']) is not bool:
        raise ValueError('deduplicate必须是布尔值')
    for key in ['rating', 'size']:
        if policy[key] not in {'drop', 'global_median', 'category_median'}:
            raise ValueError(f'{key}策略不在白名单')


def process(raw, policy, name, emit=lambda *a, **kw: None):
    validate_policy(policy)
    started = time.perf_counter()
    d = parse(raw)
    quarantine = d.loc[d.quarantined].copy()
    d = d.loc[~d.quarantined].copy()
    changes = []

    def change(rows, field, action, before, after, reason):
        for idx, r in rows.iterrows():
            changes.append({'group': name, 'source_row_id': int(r.source_row_id), 'App': r.App,
                            'field': field, 'action': action,
                            'before': str(r[before]) if before else '',
                            'after': str(d.at[idx, after]) if after else '', 'reason': reason})

    change(quarantine, 'record', 'quarantine', 'invalid_reasons', None, '核心业务范围或格式冲突；隔离并保留原始值')
    emit('tool', group=name, tool='parse_and_quarantine', input_rows=len(raw), output_rows=len(d),
         quarantined=len(quarantine), size_unit='MB; k=0.001 MB', reason='共同安全处理，不自动猜测错位修复')
    duplicates = d.loc[d.duplicated(COLUMNS, keep='first')].copy()
    if policy['deduplicate']:
        change(duplicates, 'record', 'remove_exact_duplicate', None, None, '原始13列完全相同，保留首条')
        d = d.loc[~d.duplicated(COLUMNS, keep='first')].copy()
    emit('tool', group=name, tool='exact_deduplicate', enabled=policy['deduplicate'],
         removed=len(duplicates) if policy['deduplicate'] else 0, reason='按原始13列，不按App名称')
    d['rating_value'] = d.rating_observed
    d['size_mb'] = d.size_mb_observed
    d['rating_imputed'] = False
    d['size_imputed'] = False
    d['cluster_exclusion_reason'] = ''
    imputation_stats = {}
    for key, observed, target, original in [('rating', 'rating_observed', 'rating_value', 'Rating'),
                                             ('size', 'size_mb_observed', 'size_mb', 'Size')]:
        strategy = policy[key]
        absent = d[observed].isna()
        global_median = float(d[observed].median()) if d[observed].notna().any() else None
        params = {'strategy': strategy, 'affected': int(absent.sum()), 'global_median': global_median}
        if strategy == 'drop':
            d.loc[absent, 'cluster_exclusion_reason'] += key + '_unobserved;'
        else:
            if global_median is None:
                raise ValueError(f'{key}全缺失，无法计算中位数；禁止虚构插补值')
            values = pd.Series(global_median, index=d.index)
            if strategy == 'category_median':
                groups = d.groupby('Category')[observed].agg(['median', 'count'])
                medians = groups['median'].where(groups['count'].ge(20))
                values = d.Category.map(medians).fillna(global_median)
                params['category_medians_min_count_20'] = {str(k): float(v) for k, v in medians.dropna().items()}
                params['global_fallback_rows'] = int((absent & d.Category.map(medians).isna()).sum())
            d.loc[absent, target] = values[absent]
            d.loc[absent, key + '_imputed'] = True
            change(d.loc[absent], target, strategy, original, target, '建模代理值；保留观测缺失或特殊状态，原因不推断')
        imputation_stats[key] = params
        emit('tool', group=name, tool='missing_strategy', field=key, **params)
    d['eligible_for_clustering'] = d[FEATURES].notna().all(axis=1)
    cluster = d.loc[d.eligible_for_clustering].copy()
    if not len(cluster):
        raise ValueError(f'{name}没有可用于聚类的记录')
    matrix = cluster[FEATURES].to_numpy(float)
    if not np.isfinite(matrix).all() or (matrix[:, 1:] < 0).any() or not cluster.rating_value.between(0, 5).all():
        raise ValueError('处理后特征未通过有限值、非负值或评分范围校验')
    summary = {
        'group': name, 'raw_rows': len(raw), 'quarantined_rows': len(quarantine),
        'exact_duplicates_removed': len(duplicates) if policy['deduplicate'] else 0,
        'cleaned_rows': len(d), 'cluster_rows': len(cluster),
        'cleaned_retention': len(d)/len(raw), 'cluster_retention': len(cluster)/len(raw),
        'rating_imputed': int(d.rating_imputed.sum()), 'size_imputed': int(d.size_imputed.sum()),
        'remaining_exact_duplicates': int(d.duplicated(COLUMNS).sum()),
        'observed_rating_missing': int(d.rating_observed.isna().sum()),
        'observed_size_unavailable': int(d.size_mb_observed.isna().sum()),
        'remaining_missing_all_original_columns': {c: int(missing(d[c]).sum()) for c in COLUMNS},
        'remaining_invalid_core_rows': int(d.quarantined.sum()),
        'remaining_date_failures': int(d.last_updated_iso.eq('').sum()),
        'remaining_type_price_conflicts': int(d.type_price_conflict.sum()),
        'cluster_invalid_or_missing_features': int((~np.isfinite(matrix)).any(axis=1).sum()),
        'processing_seconds': time.perf_counter()-started,
        'policy': policy, 'imputation_parameters': imputation_stats,
    }
    emit('verification', **summary)
    changes_df = pd.DataFrame(changes, columns=['group', 'source_row_id', 'App', 'field', 'action', 'before', 'after', 'reason'])
    return {'cleaned': d, 'cluster': cluster, 'quarantine': quarantine,
            'changes': changes_df, 'summary': summary}
