import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import (ROOT, find_related_groups, load_master_split,
                           load_splits, make_preprocessor)


class PreprocessingTests(unittest.TestCase):
    def test_related_components_transitive(self):
        X = pd.DataFrame({'a': [1., 2., 2., 3.]})
        c = pd.Series(['Trojan-A-ab-1.raw', 'Trojan-A-ab-2.raw',
                       'Spyware-B-cd-1.raw', 'Benign'])
        for index in [range(4), range(10, 14)]:
            with self.subTest(index=index):
                groups = find_related_groups(X.set_axis(index), c.set_axis(index))
                np.testing.assert_array_equal(groups, [0, 0, 0, 3])

    def test_train_statistics_and_preserved_extremes(self):
        p = make_preprocessor()
        self.assertFalse(hasattr(p['scaler'], 'mean_'))
        train = np.array([[0., 1.], [2., np.nan], [4., 3.]])
        p.fit(train)
        np.testing.assert_allclose(p['imputer'].statistics_, [2., 2.])
        np.testing.assert_allclose(p['scaler'].mean_, [2., 2.])
        before = p['outliers'].upper_.copy()
        incoming = np.array([[1000., np.nan]])
        imputed = p['imputer'].transform(incoming)
        self.assertTrue(p['outliers'].outlier_mask(imputed)[0, 0])
        np.testing.assert_array_equal(p['outliers'].transform(imputed), imputed)
        self.assertEqual(p.transform(incoming).shape, incoming.shape)
        np.testing.assert_array_equal(before, p['outliers'].upper_)
        np.testing.assert_allclose(p['scaler'].mean_, [2., 2.])
        self.assertFalse(hasattr(make_preprocessor()['outliers'], 'upper_'))

    def test_split_persistence_and_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            X, b, m, table = load_master_split(split_dir=tmp)
            digest = hashlib.sha256((Path(tmp) / 'master_split.csv').read_bytes()).hexdigest()
            again = load_master_split(split_dir=tmp)[3]
            pd.testing.assert_frame_equal(table, again)
            self.assertEqual(digest, hashlib.sha256((Path(tmp) / 'master_split.csv').read_bytes()).hexdigest())
            self.assertEqual(len(table), 58596)
            self.assertEqual(X.shape[1], 55)
            self.assertEqual(table.groupby('group_id').split.nunique().max(), 1)
            exact = X.groupby(list(X.columns), sort=False, dropna=False).ngroup()
            self.assertEqual(pd.DataFrame({'g': exact, 's': table.split}).groupby('g').s.nunique().max(), 1)
            binary = load_splits('binary', split_dir=tmp)
            multi = load_splits('multiclass', split_dir=tmp)
            for name in binary:
                xb, yb = binary[name]
                xm, ym = multi[name]
                pd.testing.assert_frame_equal(xb, xm)
                self.assertTrue(xb.index.equals(yb.index))
                self.assertTrue(xm.index.equals(ym.index))
                self.assertEqual(set(ym), {'Benign', 'Ransomware', 'Spyware', 'Trojan'})
            # Reject changed dataset metadata and tampered split instead of replacing.
            meta = Path(tmp) / 'metadata.json'
            saved = json.loads(meta.read_text())
            saved['raw_sha256'] = 'changed'
            meta.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'Dữ liệu hoặc cấu hình đã đổi'):
                load_master_split(split_dir=tmp)
            saved['raw_sha256'] = hashlib.sha256((ROOT / 'data/raw/Obfuscated-MalMem2022.csv').read_bytes()).hexdigest()
            meta.write_text(json.dumps(saved))
            with (Path(tmp) / 'master_split.csv').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'Checksum của file split không khớp'):
                load_master_split(split_dir=tmp)

    def test_serialization_new_process(self):
        p = make_preprocessor().fit([[0., 1.], [2., 3.], [4., 5.]])
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'preprocessor.joblib'
            joblib.dump(p, model)
            code = ('import joblib,sys; p=joblib.load(sys.argv[1]); '
                    'assert p.transform([[100.,2.]]).shape == (1,2)')
            subprocess.run([sys.executable, '-c', code, str(model)], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
