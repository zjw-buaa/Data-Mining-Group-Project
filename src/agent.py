"""Real API-only planner: validated JSON policies, never exec model code."""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from .quality import validate_policy

SYSTEM = '''你是Google Play数据质量处理计划智能体。只读质量统计和少量原始样例。
共同安全处理已固定：严格解析单位，隔离错位/非法核心值，不按数值大删除合法记录，
不凭App名称合并，不把特殊大小状态当-1或0，不推断评分缺失原因。
drop策略表示保留cleaned档案但排除该记录参与聚类；只有deduplicate移除完全重复多余记录。
只能返回一个JSON对象：{"policy":{"deduplicate":true或false,
"rating":"drop/global_median/category_median之一",
"size":"drop/global_median/category_median之一"},
"reasons":{"deduplicate":"具体理由","rating":"具体理由","size":"具体理由"}}。
category_median工具要求同分类至少20个有效观测，否则回退全局中位数。
目标是合理、可追溯的数据处理，不能声称聚类分数必然提高。不输出代码或markdown。'''


def get_plan(quality, raw, emit):
    key, base, model = [os.environ.get(k, '').strip() for k in ['LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL']]
    if not all([key, base, model]):
        return None, {'status': 'not_configured', 'real_model_called': False,
                      'missing_variable_names': [k for k in ['LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL'] if not os.environ.get(k, '').strip()],
                      'calls': 0, 'cost_usd': None, 'reason': '没有用规则或模拟输出冒充真实智能体结果'}
    parsed = urlparse(base)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query:
        return None, {'status': 'configuration_error', 'real_model_called': False, 'calls': 0,
                      'cost_usd': None, 'reason': 'LLM_BASE_URL必须为不含账号密码或查询参数的HTTPS API基础地址'}
    examples = raw.loc[raw['Rating'].isin(['', 'NaN']) | raw['Size'].eq('Varies with device')].head(4).to_dict('records')
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps({'quality': quality, 'examples': examples}, ensure_ascii=False)}]
    tokens_in = tokens_out = calls = 0
    started = time.perf_counter()
    for attempt in range(2):
        body = {'model': model, 'messages': messages, 'response_format': {'type': 'json_object'}}
        if parsed.hostname == 'api.deepseek.com':
            # DeepSeek defaults to thinking mode; this bounded planner uses non-thinking JSON.
            body.update({'thinking': {'type': 'disabled'}, 'max_tokens': 4096})
        emit('llm_request', attempt=attempt+1, endpoint=base.rstrip('/')+'/chat/completions', body=body)
        req = urllib.request.Request(base.rstrip('/')+'/chat/completions',
                                     data=json.dumps(body).encode(),
                                     headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}, method='POST')
        calls += 1
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.load(response)
            usage = data.get('usage', {})
            tokens_in += int(usage.get('prompt_tokens', 0))
            tokens_out += int(usage.get('completion_tokens', 0))
            emit('llm_response', attempt=attempt+1, response=data)
            content = data['choices'][0]['message']['content']
            plan = json.loads(content)
            if set(plan) != {'policy', 'reasons'}:
                raise ValueError('JSON顶层必须为policy/reasons')
            validate_policy(plan['policy'])
            if set(plan['reasons']) != {'deduplicate', 'rating', 'size'} or not all(isinstance(v, str) and len(v.strip()) >= 4 for v in plan['reasons'].values()):
                raise ValueError('三项策略均需具体理由')
            pin, pout = os.environ.get('LLM_INPUT_USD_PER_MILLION'), os.environ.get('LLM_OUTPUT_USD_PER_MILLION')
            cost = None
            if pin and pout and 'prompt_tokens' in usage and 'completion_tokens' in usage:
                cost = (tokens_in*float(pin)+tokens_out*float(pout))/1e6
            return plan['policy'], {'status': 'completed', 'real_model_called': True, 'calls': calls,
                                    'requested_model': model, 'returned_model': data.get('model'),
                                    'prompt_tokens': tokens_in, 'completion_tokens': tokens_out,
                                    'cost_usd': cost, 'cost_note': '未知报价或usage缺失时不估造费用',
                                    'seconds': time.perf_counter()-started, 'validated_plan': plan}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            emit('llm_validation_error', attempt=attempt+1, error=str(exc))
            messages.append({'role': 'user', 'content': '上一次响应未通过本地校验：'+str(exc)+'。请按约定重新返回JSON。'})
        except (urllib.error.URLError, TimeoutError) as exc:
            # Do not serialize Request objects, headers or provider error bodies.
            code = getattr(exc, 'code', None)
            emit('llm_transport_error', attempt=attempt+1, error_type=type(exc).__name__, http_status=code)
            return None, {'status': 'api_error', 'real_model_called': True, 'calls': calls,
                          'cost_usd': None, 'requested_model': model, 'http_status': code,
                          'reason': '真实API调用失败，未产生智能体实验结果'}
    return None, {'status': 'invalid_plan', 'real_model_called': True, 'calls': calls,
                  'prompt_tokens': tokens_in, 'completion_tokens': tokens_out, 'cost_usd': None,
                  'reason': '两次真实响应未通过白名单校验，未执行模拟计划'}
