"""The compact result bundle must still detect missing or changed evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts import replay


class FinalResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recorded = Path(self.temp.name)/'recorded'
        self.generated = Path(self.temp.name)/'generated'

    def bundle(self):
        for folder in (self.recorded, self.generated):
            for name in replay.FINAL_CSVS:
                path = folder/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('source_row_id,value,seconds\n1,2,10\n', encoding='utf-8')
            for name in replay.FINAL_JSONS:
                path = folder/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'rows': 1, 'processing_seconds': 10}), encoding='utf-8')

    def test_final_bundle_does_not_need_discarded_intermediates(self):
        self.bundle()
        # A newly generated auxiliary table need not exist in the published bundle.
        (self.generated/'common_source_row_ids.csv').write_text('source_row_id\n1\n')
        checks = replay.verify_final_results(self.generated, self.recorded)
        self.assertEqual(len(checks['csv']), len(replay.FINAL_CSVS))
        self.assertEqual(len(checks['json']), len(replay.FINAL_JSONS))

    def test_missing_required_result_cannot_silently_pass(self):
        self.bundle()
        (self.recorded/'agent/clusters.csv').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'clusters.csv'):
            replay.verify_final_results(self.generated, self.recorded)

    def test_changed_cluster_assignment_is_rejected(self):
        self.bundle()
        (self.generated/'agent/clusters.csv').write_text('source_row_id,value,seconds\n1,9,10\n')
        with self.assertRaises(AssertionError):
            replay.verify_final_results(self.generated, self.recorded)

    def test_changed_quality_count_is_rejected(self):
        self.bundle()
        (self.generated/'agent/quality_after.json').write_text('{"rows": 2, "processing_seconds": 10}')
        with self.assertRaisesRegex(AssertionError, 'quality_after.json'):
            replay.verify_final_results(self.generated, self.recorded)

    def test_only_elapsed_times_may_differ(self):
        self.bundle()
        for name in replay.FINAL_CSVS:
            (self.generated/name).write_text('source_row_id,value,seconds\n1,2,99\n')
        for name in replay.FINAL_JSONS:
            (self.generated/name).write_text('{"rows": 1, "processing_seconds": 99}')
        replay.verify_final_results(self.generated, self.recorded)


if __name__ == '__main__':
    unittest.main()
