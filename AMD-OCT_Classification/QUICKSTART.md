# Quick-start guide

This package contains **22 Google Colab notebooks** and shared Python modules. It can be deposited together with the anonymized database in a single Zenodo record.

## Getting started

1. Extract the ZIP archive on your computer.
2. In Google Colab, select **File → Upload notebook** and open `notebooks/00_dataset_inventory.ipynb`.
3. Run the first cell. When prompted, upload the complete `DMAE_Colab_Repository.zip`. This installs the shared code in the Colab session.
4. Mount your own Google Drive and set `BASE_DIR` to the directory where you will store the data and results.
5. Place the already anonymized images in `data/Normal`, `data/iORA`, `data/cORA`, `data/iRORA`, and `data/cRORA`.
6. Inspect the inventory. Continue with training and extraction if you need new features. If a verified feature NPZ is already available, open notebook 03 or an individual classifier notebook directly.

For training and neural classifiers, select **Runtime → Change runtime type → GPU**. CNN training starts only after `RUN_TRAINING` is set to `True`.

## Using existing feature caches

Copy `DB_CNNpropia.npz` or `DB_VGG16.npz` into `BASE_DIR/features`. Each NPZ must contain `features` and `labels`. If `class_names` metadata are present, set `CLASS_NAMES = None` to read the stored order. For the two historical caches reviewed, the verified label order is **Normal, cORA, cRORA, iORA, iRORA**. Do not change numeric labels merely to match the presentation order in the manuscript tables.

For ResNet50 and EfficientNet, read `class_names` from the NPZ or explicitly supply a verified class order. The code does not infer the order of a different legacy cache.

## Preserved methods and organizational changes

The verified classifier definitions and historical image-level evaluation protocol are retained. Paths, metrics and exports are centralized; output labels use the correct anatomical categories, and all stored notebook outputs have been removed. ROC curves are computed from predictions and scores, never reconstructed from an isolated AUC value.

New experiment outputs are saved in separate directories. The `results/reported` directory contains the historical manuscript results and is not overwritten.

The supplied archive does not contain a standalone VGG16 ensemble notebook or the original Dense256 CNN weights. The Dense32 development model is not substituted for them. Code to train a new Dense256 CNN is included; its outputs belong to a new experiment. The main README explains the scope of each workflow.

## Depositing the package

Attach this code ZIP, the anonymized database and the licensing terms agreed by the authors and institution to the same Zenodo record. A second repository is not required. The package does not upload or publish files automatically.
