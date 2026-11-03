"""Conservative, deterministic checks. No claim of general semantic correctness."""
import re
from src.quality import validate_policy
VERSION = 'reason-rules-1.0'
LIMITATION = '有限规则检查；未发现冲突不等于自然语言理由完整、真实或科学正确。'

def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'policy', 'reasons'}:
        raise ValueError('JSON顶层必须为policy/reasons对象')
    if not isinstance(plan['policy'], dict):
        raise ValueError('policy必须为对象')
    validate_policy(plan['policy'])
    reasons = plan['reasons']
    if not isinstance(reasons, dict) or set(reasons) != {'deduplicate','rating','size'}:
        raise ValueError('reasons必须为包含三项具体理由的对象')
    if not all(isinstance(v,str) and len(v.strip()) >= 4 for v in reasons.values()):
        raise ValueError('三项策略均需具体理由')

# Split at sentence/clause boundaries: do not let a negation leak into the next action.
NEGATIVE_MEDIAN = re.compile(r'(?:不适合|不适宜|不宜|不应|不能|不可|不使用|不采用|不做|不建议|无需|避免)[^，。；;\n]{0,24}中位数(?:填[补充]|插补)?', re.I)
EN_NEGATIVE_MEDIAN = re.compile(r'(?:do\s+not|not\s+suitable\s+(?:for|to)|avoid|cannot|should\s+not|without)\s+(?:\w+[ -]+){0,5}(?:(?:global|category|categorical)\s+)?median(?:\s+imputation)?',re.I)
POSITIVE_IMPUTE = re.compile(r'(?:采用|使用|用|通过|按|取|进行|实施)[^，。；;\n]{0,20}中位数[^，。；;\n]{0,12}(?:填[补充]|插补|代理|建模)|(?:填[补充]|插补)[^，。；;\n]{0,12}中位数|按(?:类别|分类|同类)(?:取)?中位数|(?:impute|fill|replace|use)\b[^.;\n]{0,45}\bmedian',re.I)
DROP = re.compile(r'(?:不填[补充]|不插补|排除[^，。；;\n]{0,12}聚类|不参与聚类|不参加聚类|退出聚类|exclude[^.;\n]{0,35}cluster|drop\b|聚类[^，。；;\n]{0,6}排除|no\s+imputation)',re.I)
DEDUPE_YES = re.compile(r'(?:移除|删除|去除|清除)[^，。；;\n]{0,14}(?:重复|副本|多余记录|多余行)|(?:remove|delete)\b[^.;\n]{0,30}(?:duplicat|copies)|(?:移除|删除)它们|执行去重|进行去重',re.I)
DEDUPE_NO = re.compile(r'(?:不(?:进行|执行)?去重|保留[^，。；;\n]{0,12}(?:重复|副本)|不(?:移除|删除)[^，。；;\n]{0,12}重复|keep[^.;\n]{0,30}duplicat|do\s+not\s+deduplicat)',re.I)

def _positive(text, pattern):
    """Ignore positive-looking substrings inside an explicit negated clause."""
    found=[]
    for match in pattern.finditer(text):
        prefix=text[max(0,match.start()-8):match.start()]
        if re.search(r'(?:不|未|无需|避免|not\s|never\s)[^，。；;\n]{0,5}$',prefix,re.I):
            continue
        found.append(match.group(0))
    return found

def _median_scope(fragment):
    if re.search(r'全局|全表|global',fragment,re.I): return 'global_median'
    if re.search(r'类别|分类|同类|category|categorical',fragment,re.I): return 'category_median'
    return 'any_median'

def audit_plan(plan):
    validate_plan(plan)
    issues=[];fields={}
    def issue(field,code,evidence):
        issues.append({'field':field,'code':code,'evidence':evidence})
    for field,chosen in plan['policy'].items():
        text=plan['reasons'][field].strip()
        if field=='deduplicate':
            yes=_positive(text,DEDUPE_YES);no=[m.group(0) for m in DEDUPE_NO.finditer(text)]
            if chosen and no: issue(field,'policy_reason_conflict',no[0])
            if not chosen and yes: issue(field,'policy_reason_conflict',yes[0])
            if yes and no: issue(field,'reason_internal_conflict','同时出现保留与删除重复项的描述')
            if not (yes if chosen else no): issue(field,'unrecognized_reason','未识别到与去重选择一致的明确动作')
        else:
            negatives=[m.group(0) for regex in [NEGATIVE_MEDIAN,EN_NEGATIVE_MEDIAN] for m in regex.finditer(text)]
            positives=_positive(text,POSITIVE_IMPUTE)
            drops=bool(DROP.search(text))
            if chosen!='drop':
                if re.search(r'(?:不填补|不填充|不插补|不能填补|不能填充|不做插补|no\s+imputation)',text,re.I):
                    issue(field,'policy_reason_conflict','理由同时否定填补，但策略选择中位数填补')
                contrary=[s for s in negatives if _median_scope(s) in ('any_median',chosen)]
                if contrary: issue(field,'policy_reason_conflict',contrary[0])
                if not positives: issue(field,'unrecognized_reason','未识别到明确的中位数填补/建模动作')
                # category strategies may explicitly fall back to a global median.
                if chosen=='global_median' and re.search(r'(?:按|用|采用)[^，。；;\n]{0,6}(?:类别|分类|同类)中位数',text):
                    issue(field,'policy_reason_conflict','全局策略的理由却选择类别中位数')
                if re.search(r'(?:不参与聚类|不参加聚类|排除[^，。；;\n]{0,8}聚类)',text) and not re.search(r'非法|错位|异常|隔离',text):
                    issue(field,'ambiguous_scope','选择填补，却同时描述排除聚类；需澄清适用记录')
            else:
                if positives: issue(field,'policy_reason_conflict',positives[0])
                if not drops: issue(field,'unrecognized_reason','未识别到排除聚类或不填补动作')
        fields[field]={'policy':chosen,'reason':text}
    return {'version':VERSION,'status':'needs_clarification' if issues else 'no_rule_conflict_found',
            'allow_execution':not issues,'issues':issues,'fields':fields,'limitation':LIMITATION}