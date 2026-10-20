"""Classifier settings transcribed from the supplied cached-feature notebooks."""
import numpy as np

AVAILABLE = {
    'custom_cnn': ('svm', 'knn3', 'random_forest', 'xgboost', 'gbm', 'feedforward', 'rnn', 'ensemble'),
    'vgg16': ('svm', 'knn3', 'random_forest', 'xgboost', 'gbm', 'feedforward', 'rnn'),
    'resnet50': ('knn3',), 'efficientnet': ('knn3',)}

def fit_predict(extractor, classifier, x_train, y_train, x_valid, y_valid):
    if classifier not in AVAILABLE.get(extractor, ()):
        raise ValueError(f'No verified supplied implementation for {extractor}/{classifier}. Available: {AVAILABLE}')
    if set(np.unique(y_train)) != set(range(5)):
        raise ValueError('Training partition must contain all five categories.')
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    custom = extractor == 'custom_cnn'
    model = None
    if classifier == 'svm':
        model = SVC(kernel='rbf', probability=not custom)
    elif classifier == 'knn3':
        model = KNeighborsClassifier(n_neighbors=3)
    elif classifier == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42 if custom else None)
    elif classifier == 'gbm':
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=3)
    elif classifier == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(eval_metric='logloss' if custom else 'mlogloss', random_state=42 if custom else None)
    elif classifier == 'feedforward' and not custom:
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    if model is not None:
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        scores = model.decision_function(x_valid) if classifier == 'svm' and custom else model.predict_proba(x_valid)
        ordered_scores = np.empty((len(x_valid), 5))
        for column, label in enumerate(model.classes_):
            ordered_scores[:, int(label)] = scores[:, column]
        return pred, ordered_scores
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers
    tf.keras.backend.clear_session()
    # Neural classifiers used their default float32 policy, independent of extractor training.
    tf.keras.mixed_precision.set_global_policy('float32')
    if classifier == 'feedforward':
        model = Sequential([layers.Input((x_train.shape[1],)), layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(), layers.Dropout(.5), layers.Dense(512, activation='relu'),
            layers.BatchNormalization(), layers.Dropout(.5), layers.Dense(256, activation='relu'),
            layers.BatchNormalization(), layers.Dense(128, activation='relu'), layers.Dense(5, activation='softmax')])
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_valid, y_valid), verbose=1)
        scores = model.predict(x_valid, verbose=0)
    elif classifier == 'rnn':
        x_train, x_valid = x_train[:, None, :], x_valid[:, None, :]
        blocks = [layers.Input((1, x_train.shape[2])), layers.SimpleRNN(64, activation='relu')]
        if custom: blocks.append(layers.Dropout(.4))
        blocks.extend([layers.Dense(32, activation='relu'), layers.Dense(5, activation='softmax')])
        model = Sequential(blocks)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=30 if custom else 50, batch_size=16 if custom else 32,
                  validation_data=(x_valid, y_valid), verbose=1)
        scores = model.predict(x_valid, verbose=0)
    elif classifier == 'ensemble':
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import VotingClassifier
        model = Sequential([layers.Input((x_train.shape[1], 1)), layers.SimpleRNN(128, activation='tanh'),
            layers.BatchNormalization(), layers.Dropout(.5), layers.Dense(5, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train[:, :, None], y_train, epochs=30, batch_size=32,
                  validation_data=(x_valid[:, :, None], y_valid), verbose=1)
        recurrent = model.predict(x_valid[:, :, None], verbose=0)
        voting = VotingClassifier(estimators=[('svm', SVC(probability=True, kernel='rbf')),
            ('ffnn', MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', solver='adam', max_iter=300))], voting='soft')
        voting.fit(x_train, y_train)
        scores = .5 * voting.predict_proba(x_valid) + .5 * recurrent
    else:
        raise ValueError(classifier)
    return np.argmax(scores, axis=1), scores
