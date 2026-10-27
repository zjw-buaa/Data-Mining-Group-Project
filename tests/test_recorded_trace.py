import json
from pathlib import Path
import tempfile
import unittest

from replay_recorded_run import DEFAULT_RECORDED, ROOT, load_trace


class TraceProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_base = (ROOT/'.cache/trace_tests').resolve()
        cls.temp_base.mkdir(parents=True, exist_ok=True)

    def make_copy(self, temp):
        target = Path(temp)
        if not target.resolve().is_relative_to(self.temp_base):
            raise ValueError('测试副本必须在工作区临时目录内')
        for name in ['agent_status.json', 'run_manifest.json', 'quality_before.json', 'logs/pipeline.jsonl']:
            p = target/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((DEFAULT_RECORDED/name).read_bytes())
        return target

    def test_rejects_non_real_model_status(self):
        with tempfile.TemporaryDirectory(dir=self.temp_base) as temp:
            path = self.make_copy(temp)
            status = json.loads((path/'agent_status.json').read_text(encoding='utf-8'))
            status['real_model_called'] = False
            (path/'agent_status.json').write_text(json.dumps(status), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '真实模型'):
                load_trace(path)

    def test_rejects_plan_not_matching_original_response(self):
        with tempfile.TemporaryDirectory(dir=self.temp_base) as temp:
            path = self.make_copy(temp)
            status = json.loads((path/'agent_status.json').read_text(encoding='utf-8'))
            status['validated_plan']['policy']['deduplicate'] = False
            (path/'agent_status.json').write_text(json.dumps(status), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '原始模型响应'):
                load_trace(path)


if __name__ == '__main__':
    unittest.main()
