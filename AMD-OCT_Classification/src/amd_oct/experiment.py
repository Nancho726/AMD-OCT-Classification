"""Repeated image-level holdouts, saved partitions and labeled diagnostic outputs."""
from pathlib import Path
import importlib.metadata
import platform
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import load_cache, sha256, write_json, DISPLAY_CLASS_ORDER
from .metrics import evaluate, save_figures
from .models import fit_predict

def environment():
    versions = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'tensorflow', 'keras', 'xgboost', 'pillow']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return versions

def run_experiment(cache_path, output_dir, extractor='custom_cnn', classifier='svm',
                   repetitions=5, split_seed=None, class_names=None):
    """Historical image-level protocol; no stratification or patient grouping is introduced."""
    if repetitions < 1: raise ValueError('repetitions must be positive.')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory is not empty; use a fresh run name to preserve results.')
    expected = {'custom_cnn': 256, 'vgg16': 512}.get(extractor)
    x, y, names = load_cache(cache_path, class_names, expected)
    output.mkdir(parents=True, exist_ok=True)
    x = x.astype(np.float32, copy=False)
    write_json(output / 'run_config.json', {'extractor': extractor, 'classifier': classifier,
        'repetitions': repetitions, 'test_size': .2, 'split_seed': split_seed, 'stratify': None,
        'partition_unit': 'image', 'feature_extractor_refit_inside_split': False,
        'cache_sha256': sha256(cache_path), 'cache_shape': list(x.shape), 'class_names': names,
        'observed_class_counts': {name: int(np.sum(y == i)) for i, name in enumerate(names)},
        'environment': environment(), 'protocol': 'repeated image-level holdout on cached features'})
    records = []
    for run in range(repetitions):
        folder = output / f'run_{run + 1:02d}'; folder.mkdir()
        seed = None if split_seed is None else int(split_seed) + run
        train, valid = train_test_split(np.arange(len(y)), test_size=.2, random_state=seed)
        np.savez_compressed(folder / 'partition_indices.npz', train=train, validation=valid)
        start = time.perf_counter()
        pred, scores = fit_predict(extractor, classifier, x[train], y[train], x[valid], y[valid])
        elapsed = time.perf_counter() - start
        metrics, report, cm, roc = evaluate(y[valid], pred, scores, names)
        write_json(folder / 'metrics.json', metrics); write_json(folder / 'classification_report.json', report)
        write_json(folder / 'roc.json', roc)
        np.savez_compressed(folder / 'predictions.npz', validation_indices=valid, y_true=y[valid],
                            y_pred=pred, scores=scores, class_names=np.asarray(names))
        pd.DataFrame(cm, index=DISPLAY_CLASS_ORDER, columns=DISPLAY_CLASS_ORDER).to_csv(folder / 'confusion_matrix.csv')
        save_figures(folder, cm, roc)
        records.append({'run': run + 1, 'extractor': extractor, 'classifier': classifier,
                        'fit_and_prediction_seconds': elapsed, **metrics})
        pd.DataFrame(records).to_csv(output / 'runs.csv', index=False)
    runs = pd.DataFrame(records)
    columns = ['accuracy', 'weighted_f1', 'balanced_accuracy', 'macro_specificity', 'macro_auc']
    summary = runs[columns].agg(['mean', 'std']).T
    summary.index.name = 'metric'; summary.to_csv(output / 'summary.csv')
    return runs, summary
