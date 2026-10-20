"""Explicit class mappings, image inventory, and safe feature-cache access."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image

CACHE_CLASS_ORDER = ('Normal', 'cORA', 'cRORA', 'iORA', 'iRORA')
DISPLAY_CLASS_ORDER = ('Normal', 'iORA', 'cORA', 'iRORA', 'cRORA')
TRAINING_CLASS_ORDER = DISPLAY_CLASS_ORDER  # Explicit order in the original CNN training generator.
EXPECTED_COUNTS = {'Normal': 4406, 'iORA': 603, 'cORA': 683, 'iRORA': 653, 'cRORA': 976}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def inventory_images(dataset_dir, output_csv=None):
    """Record relative paths only; requires an already anonymized dataset."""
    root = Path(dataset_dir)
    missing = [c for c in CACHE_CLASS_ORDER if not (root / c).is_dir()]
    if missing:
        raise FileNotFoundError(f'Missing class folders: {missing}; expected under {root}')
    rows = []
    for label, name in enumerate(CACHE_CLASS_ORDER):
        for path in sorted((root / name).rglob('*')):
            if path.is_file() and path.suffix.lower() == '.png':
                with Image.open(path) as image:
                    width, height = image.size
                    rows.append({'image_path': path.relative_to(root).as_posix(), 'category': name,
                                 'label': label, 'width': width, 'height': height})
    result = pd.DataFrame(rows, columns=['image_path', 'category', 'label', 'width', 'height'])
    if result.empty:
        raise ValueError('No PNG images found.')
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result

def load_cache(path, class_names=None, expected_dimension=None):
    """Read numeric NPZ caches without pickle and never guess class order from counts."""
    with np.load(path, allow_pickle=False) as data:
        if not {'features', 'labels'} <= set(data.files):
            raise ValueError('NPZ must contain features and labels.')
        features = np.asarray(data['features'])
        labels = np.asarray(data['labels'])
        stored_names = tuple(str(x) for x in data['class_names']) if 'class_names' in data else None
    if labels.ndim != 1 or features.ndim < 2 or features.shape[0] != len(labels):
        raise ValueError('Expected features (N, ...) and labels (N,) with matching N.')
    if not np.issubdtype(features.dtype, np.number) or np.iscomplexobj(features):
        raise ValueError('Features must be real numeric arrays.')
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Labels must be finite integers.')
    labels = labels.astype(np.int64)
    if stored_names is None and class_names is None:
        raise ValueError('Legacy cache has no class_names. Supply its verified label order explicitly.')
    names = tuple(class_names) if class_names is not None else stored_names
    if stored_names is not None and class_names is not None and stored_names != names:
        raise ValueError('Supplied class names conflict with NPZ metadata.')
    if len(names) != 5 or len(set(names)) != 5 or set(names) != set(CACHE_CLASS_ORDER):
        raise ValueError('class_names must contain the five distinct anatomical categories.')
    if np.any(labels < 0) or np.any(labels >= 5):
        raise ValueError('Labels must be between 0 and 4.')
    features = features.reshape(len(features), -1)
    if expected_dimension is not None and features.shape[1] != expected_dimension:
        raise ValueError(f'Expected {expected_dimension} features; received {features.shape[1]}.')
    if not np.isfinite(features).all():
        raise ValueError('Features contain NaN or infinity.')
    return features, labels, names

def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
