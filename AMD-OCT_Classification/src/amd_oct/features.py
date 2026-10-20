"""Batch extraction with explicit label metadata and 256/512 dimension checks."""
from pathlib import Path
import numpy as np
from .data import CACHE_CLASS_ORDER, write_json

def extract_features(dataset_dir, output_path, extractor='custom_cnn', model_path=None):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import Model
    if extractor not in ('custom_cnn', 'vgg16'):
        raise ValueError('Exploratory backbones are evaluated from their supplied caches only.')
    out = Path(output_path)
    if out.exists(): raise FileExistsError('Feature cache exists; choose a new filename.')
    out.parent.mkdir(parents=True, exist_ok=True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    if extractor == 'custom_cnn':
        if model_path is None or not Path(model_path).is_file():
            raise FileNotFoundError('Supply a 256-component trained CNN or run the training notebook first.')
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
        candidates = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense) and layer.units == 256]
        if len(candidates) != 1:
            raise ValueError('Expected one Dense(256) feature layer. Dense32 development models are incompatible.')
        feature_model = Model(model.inputs, candidates[0].output)
        dimension = 256
    else:
        # Frozen base + average pooling matches the 512-component inference representation.
        # Historical cache extraction rescaled by 1/255; do not silently replace with preprocess_input.
        feature_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                                    input_shape=(384, 384, 3), pooling='avg')
        feature_model.trainable = False
        dimension = 512
    generator = ImageDataGenerator(rescale=1./255).flow_from_directory(str(dataset_dir),
        target_size=(384, 384), batch_size=32, classes=list(CACHE_CLASS_ORDER),
        class_mode='sparse', shuffle=False, color_mode='rgb')
    if generator.samples == 0: raise ValueError('No supported images found.')
    batches = []
    for i in range(len(generator)):
        images, _ = generator[i]
        values = np.asarray(feature_model.predict(images, verbose=0)).reshape(len(images), -1)
        if values.shape[1] != dimension: raise ValueError('Unexpected feature dimension.')
        batches.append(values.astype(np.float16))
    features = np.concatenate(batches)
    np.savez_compressed(out, features=features, labels=generator.classes.astype(np.int64),
        class_names=np.asarray(CACHE_CLASS_ORDER), image_paths=np.asarray(generator.filenames))
    write_json(out.with_suffix('.json'), {'extractor': extractor, 'shape': list(features.shape),
        'class_names': CACHE_CLASS_ORDER, 'input_size': [384, 384], 'rescale': 1./255,
        'shuffle': False, 'model_source': 'provided or newly trained model' if extractor == 'custom_cnn' else 'Keras ImageNet VGG16 frozen base',
        'precision': 'float16 storage'})
    return out
