import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from amd_oct.data import CACHE_CLASS_ORDER, DISPLAY_CLASS_ORDER, load_cache, inventory_images
from amd_oct.metrics import evaluate
from amd_oct.models import fit_predict
from amd_oct.experiment import run_experiment

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def cache(self, **overrides):
        values = {'features': np.ones((10, 256)), 'labels': np.arange(10) % 5,
                  'class_names': np.array(CACHE_CLASS_ORDER)}
        values.update(overrides); path = self.root / 'cache.npz'
        np.savez_compressed(path, **values); return path
    def test_cache_dimensions_and_metadata(self):
        x, y, names = load_cache(self.cache(), expected_dimension=256)
        self.assertEqual(x.shape, (10, 256)); self.assertEqual(names, CACHE_CLASS_ORDER)
    def test_legacy_requires_mapping(self):
        path = self.root / 'legacy.npz'
        np.savez(path, features=np.ones((10, 256)), labels=np.arange(10) % 5)
        with self.assertRaises(ValueError): load_cache(path)
        self.assertEqual(load_cache(path, CACHE_CLASS_ORDER)[2], CACHE_CLASS_ORDER)
    def test_metadata_conflict_rejected(self):
        with self.assertRaises(ValueError): load_cache(self.cache(), DISPLAY_CLASS_ORDER)
    def test_bad_feature_dimension_rejected(self):
        with self.assertRaises(ValueError): load_cache(self.cache(features=np.ones((10, 32))), expected_dimension=256)
    def test_nonfinite_features_rejected(self):
        with self.assertRaises(ValueError): load_cache(self.cache(features=np.full((10, 256), np.nan)))
    def test_nonintegral_and_out_of_range_labels_rejected(self):
        for values in [np.ones(10)*.5, np.ones(10)*5]:
            with self.assertRaises(ValueError): load_cache(self.cache(labels=values))
    def test_display_reorders_both_axes_and_names(self):
        true = np.repeat(np.arange(5), [10, 2, 3, 4, 5]); scores = np.eye(5)[true]
        metrics, report, cm, roc = evaluate(true, true, scores, CACHE_CLASS_ORDER)
        self.assertEqual(cm.diagonal().tolist(), [10, 4, 2, 5, 3])
        self.assertEqual(report['iORA']['support'], 4)
        self.assertEqual(metrics['macro_auc'], 1.)
        self.assertEqual(metrics['macro_specificity'], 1.)
        self.assertEqual(metrics['exact_subtype_among_atrophy'], 1.)
    def test_missing_class_roc_is_explicit(self):
        true=np.arange(4); scores=np.eye(5)[true]
        metrics, _, _, roc=evaluate(true,true,scores,CACHE_CLASS_ORDER)
        self.assertIsNone(metrics['macro_auc']); self.assertIsNone(roc['iRORA']['auc'])
    def test_inventory_counts_and_relative_paths(self):
        for category in CACHE_CLASS_ORDER:
            (self.root/category).mkdir()
            Image.new('L',(938,625)).save(self.root/category/'001.png')
        inventory=inventory_images(self.root)
        self.assertEqual(len(inventory),5);self.assertTrue((inventory.width==938).all())
        self.assertTrue(all(not Path(x).is_absolute() for x in inventory.image_path))
    def test_vgg_ensemble_not_fabricated(self):
        with self.assertRaises(ValueError): fit_predict('vgg16','ensemble',np.zeros((10,4)),np.arange(10)%5,np.zeros((5,4)),np.arange(5))
    def test_experiment_records_indices_and_predictions(self):
        rng=np.random.default_rng(7);labels=np.tile(np.arange(5),30)
        features=rng.normal(size=(150,256));features[:,0:5]+=np.eye(5)[labels]*5
        path=self.root/'features.npz';np.savez_compressed(path,features=features,labels=labels,class_names=np.array(CACHE_CLASS_ORDER))
        out=self.root/'result'
        runs,summary=run_experiment(path,out,'custom_cnn','svm',repetitions=2,split_seed=42)
        self.assertEqual(len(runs),2)
        with np.load(out/'run_01/partition_indices.npz') as part:
            self.assertEqual(len(set(part['train']) & set(part['validation'])),0)
            self.assertEqual(len(part['train'])+len(part['validation']),150)
        self.assertTrue((out/'run_01/roc.json').exists())
        with self.assertRaises(FileExistsError):run_experiment(path,out,'custom_cnn','svm',1,42)

if __name__ == '__main__': unittest.main()
