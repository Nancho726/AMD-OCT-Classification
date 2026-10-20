"""Source-derived 256-component custom CNN; training creates a new experiment."""
from pathlib import Path
from .data import TRAINING_CLASS_ORDER, write_json

def build_custom_cnn():
    import tensorflow as tf
    from tensorflow.keras import layers
    model = tf.keras.Sequential(name='custom_cnn_256')
    model.add(layers.Input((384, 384, 3)))
    for index, filters in enumerate([16, 32, 64, 128, 128], 1):
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='valid', name=f'conv_{index}'))
        model.add(layers.BatchNormalization(name=f'bn_{index}'))
        model.add(layers.MaxPooling2D((2, 2), name=f'pool_{index}'))
    model.add(layers.Flatten(name='flatten_12800'))
    model.add(layers.Dense(256, activation='relu', name='dense_features'))
    model.add(layers.Dropout(.4))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_custom_cnn(dataset_dir, output_dir, epochs=30):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    import pandas as pd
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use an empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    generator = ImageDataGenerator(validation_split=.2, rescale=1./255)
    train = generator.flow_from_directory(str(dataset_dir), target_size=(384, 384), batch_size=32,
        classes=list(TRAINING_CLASS_ORDER), class_mode='sparse', subset='training', shuffle=True)
    valid = generator.flow_from_directory(str(dataset_dir), target_size=(384, 384), batch_size=32,
        classes=list(TRAINING_CLASS_ORDER), class_mode='sparse', subset='validation', shuffle=False)
    if not train.samples or not valid.samples: raise ValueError('Both image subsets must be nonempty.')
    model = build_custom_cnn()
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(str(out / 'best_val_accuracy.keras'), monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=5, min_lr=1e-7)]
    history = model.fit(train, validation_data=valid, epochs=epochs, callbacks=callbacks)
    model.save(out / 'custom_cnn_256.keras')
    pd.DataFrame(history.history).to_csv(out / 'history.csv', index=False)
    write_json(out / 'training_config.json', {'class_names': TRAINING_CLASS_ORDER, 'input_size': [384, 384, 3],
        'feature_components': 256, 'flatten_components': 12800, 'batch_size': 32, 'max_epochs': epochs,
        'rescale': 1./255, 'augmentation': False, 'class_weight': None,
        'selection': 'restored validation-loss weights; separate accuracy checkpoint',
        'validation_shuffle': False, 'note': 'new training run; not the original cached-feature weights'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ['loss', 'accuracy']):
        ax.plot(history.history[key], label='Training'); ax.plot(history.history['val_' + key], label='Validation')
        ax.set(xlabel='Epoch', ylabel=key); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=180); plt.close(fig)
    return model
