"""Real API-only planner: validated JSON policies, never exec model code."""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from src.quality import validate_policy
from src.agent import SYSTEM as ORIGINAL_SYSTEM
from extensions.reason_guard import validate_plan, audit_plan, VERSION
from pathlib import Path
import hashlib

SYSTEM = ORIGINAL_SYSTEM + '\n请逐项说明理由与policy动作的对应关系。不得一面否定填补，一面选择填补；若否定全局中位数但采用类别中位数，必须明确区分。无法说明时返回保守一致的策略。'

class ReasonConflict(ValueError):
    pass

def provenance():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root/'run.py', root/'extensions/guarded_agent.py', root/'extensions/reason_guard.py']}



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
    last_audit = None
    emit('reason_guard_started', version=VERSION, source_sha256=provenance(), scope='有限规则，不等于完整语义验证')
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
            if not isinstance(content,str):
                raise ValueError('message.content必须为文本')
            messages.append({'role':'assistant', 'content':content})
            plan = json.loads(content)
            validate_plan(plan)
            last_audit = audit_plan(plan)
            emit('llm_reason_audit', attempt=attempt+1, **last_audit)
            if not last_audit['allow_execution']:
                raise ReasonConflict(json.dumps(last_audit['issues'], ensure_ascii=False))
            pin, pout = os.environ.get('LLM_INPUT_USD_PER_MILLION'), os.environ.get('LLM_OUTPUT_USD_PER_MILLION')
            cost = None
            if pin and pout and 'prompt_tokens' in usage and 'completion_tokens' in usage:
                cost = (tokens_in*float(pin)+tokens_out*float(pout))/1e6
            return plan['policy'], {'status': 'completed', 'real_model_called': True, 'calls': calls,
                                    'requested_model': model, 'returned_model': data.get('model'),
                                    'prompt_tokens': tokens_in, 'completion_tokens': tokens_out,
                                    'cost_usd': cost, 'cost_note': '按环境配置的统一输入输出单价估算，非实际账单；未做缓存差异计价，精确费用需账单核验',
                                    'seconds': time.perf_counter()-started, 'validated_plan': plan,
                                    'reason_guard':last_audit, 'enhancement_source_sha256':provenance()}
        except ReasonConflict as exc:
            emit('llm_reason_conflict', attempt=attempt+1, error=str(exc), action='retry' if attempt==0 else 'stop')
            messages.append({'role':'user','content':'理由与策略检查未通过：'+str(exc)+'。请澄清矛盾并返回完整JSON，不得仅改写理由来掩盖不一致。'})
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
                  'reason': '两次响应仍未通过结构或理由规则检查，智能体计划未执行',
                  'reason_guard':last_audit, 'enhancement_source_sha256':provenance()}
