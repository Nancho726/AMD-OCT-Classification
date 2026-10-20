# Validation of the organized code release

Performed on 16 October 2026.

- Parsed all **22 notebooks** and compiled the syntax of **103 code cells**.
- Confirmed empty outputs and null execution counters in every executable notebook.
- Parsed all eight Python modules/test files.
- Passed **11 unit/integration checks**, including explicit cache mappings, rejection of conflicting metadata, rejection of Dense32-sized caches in a Dense256 path, both-axis anatomical matrix reordering, class-specific ROC handling, PNG inventory and saved hold-out partitions.
- Executed one complete SVM hold-out workflow on each of the supplied custom CNN and VGG16 feature caches. Both completed and produced partition indices, predictions, metrics, confusion matrices and ROC files. These are **new smoke-test runs**, not replacements for the manuscript results; their generated outputs are not bundled with the reported results.
- Preserved the original 90-row reported-results CSV unchanged.
- Confirmed the release contains no NPZ feature caches, fitted Keras models or patient images.

Validation environment: Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; matplotlib 3.10.8; Pillow 12.3.0. `validation_summary.json` records the environment mechanically.

TensorFlow, Keras and XGBoost were not installed in this validation environment. Their source-derived paths were syntax-checked but not executed here. No neural retraining or authenticated Google Colab session was performed. Dependency version ranges are compatibility targets, not reconstructed historical versions. Dataset-dependent training and extraction require the anonymized data in the user's Colab/Drive environment.

Exploratory feature caches can require substantial RAM, particularly if they contain flattened spatial maps. These paths do not imply that an arbitrary several-gigabyte cache will fit a standard Colab runtime.
