import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.quality import COLUMNS, read_raw, parse, process, FIXED, MINIMUM, validate_policy
from src.agent import get_plan
from src.pipeline import ROOT


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_raw(ROOT/'data/raw/googleplaystore.csv')

    def test_raw_counts_and_known_anomaly(self):
        d = parse(self.raw)
        self.assertEqual(len(d), 10841)
        self.assertEqual(int(d.rating_missing.sum()), 1474)
        self.assertEqual(int(d.exact_duplicate_extra.sum()), 483)
        self.assertEqual(d.loc[d.quarantined, 'source_row_id'].tolist(), [10473])
        self.assertEqual(d.loc[d.quarantined, 'Rating'].iloc[0], '19')

    def test_units_and_special_state(self):
        fixture = self.raw.iloc[:3].copy()
        fixture['Size'] = ['19M', '201k', 'Varies with device']
        d = parse(fixture)
        np.testing.assert_allclose(d.size_mb_observed.iloc[:2], [19., .201])
        self.assertTrue(np.isnan(d.size_mb_observed.iloc[2]))
        self.assertEqual(d.size_state.iloc[2], 'varies')

    def test_exact_duplicate_ignores_id_but_not_other_columns(self):
        fixture = self.raw.iloc[[0, 0, 0]].copy().reset_index(drop=True)
        fixture.source_row_id = [1, 2, 3]
        fixture.loc[2, 'Reviews'] = '999'
        g = process(fixture, FIXED, 'fixture')
        self.assertEqual(g['cleaned'].source_row_id.tolist(), [1, 3])

    def test_original_not_mutated_and_observation_not_fabricated(self):
        original = self.raw.copy(deep=True)
        g = process(self.raw, FIXED, 'fixed')
        pd.testing.assert_frame_equal(original, self.raw)
        d = g['cleaned']
        self.assertEqual(len(d), 10357)
        self.assertEqual(int(d.duplicated(COLUMNS).sum()), 0)
        self.assertTrue(d.loc[d.rating_missing, 'rating_observed'].isna().all())
        self.assertTrue(d.loc[d.size_varies, 'size_mb_observed'].isna().all())
        self.assertTrue(d.loc[d.size_varies, 'size_imputed'].all())
        self.assertFalse((d.size_mb == -1).any())

    def test_minimum_keeps_duplicates_and_no_imputation(self):
        d = process(self.raw, MINIMUM, 'minimum')
        self.assertEqual(d['summary']['remaining_exact_duplicates'], 483)
        self.assertEqual(d['summary']['rating_imputed'], 0)
        self.assertTrue(d['cluster'].rating_observed.notna().all())
        self.assertTrue(d['cluster'].size_mb_observed.notna().all())

    def test_legitimate_popular_and_expensive_apps_retained(self):
        p = parse(self.raw)
        ids = set(p.loc[~p.quarantined & ~p.exact_duplicate_extra].nlargest(10, 'installs_value').source_row_id)
        g = process(self.raw, FIXED, 'fixed')
        self.assertTrue(ids.issubset(set(g['cleaned'].source_row_id)))

    def test_invalid_policy_cannot_execute_code(self):
        with self.assertRaises(ValueError):
            validate_policy({'deduplicate': True, 'rating': '__import__("os")', 'size': 'global_median'})
        with self.assertRaises(ValueError):
            validate_policy({'deduplicate': 'yes', 'rating': 'drop', 'size': 'drop'})

    def test_no_api_config_is_not_a_fake_agent(self):
        with patch.dict(os.environ, {}, clear=True):
            plan, status = get_plan({}, self.raw, lambda *a, **k: None)
        self.assertIsNone(plan)
        self.assertFalse(status['real_model_called'])
        self.assertEqual(status['status'], 'not_configured')


if __name__ == '__main__':
    unittest.main()
