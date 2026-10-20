# Five-category OCT atrophy classification — Colab research code

Companion code for **Hybrid learning for OCT-based classification of retinal atrophy in age-related macular degeneration**.

**Authors:** Andrés Felipe Quiñones Lucio — Instituto para Niños Ciegos y Sordos, Cali, Colombia; Hernán Darío Vargas Cardona — Pontificia Universidad Javeriana Cali, Cali, Colombia.

Contact: andres.quinones@ciegosysordos.org.co; corresponding author: hernan.vargas@javerianacali.edu.co.

See `QUICKSTART.md` for concise setup instructions. All user-facing instructions are in English.

## Start in Google Colab

1. Extract `DMAE_Colab_Repository.zip` on your computer and open one of the notebooks in Colab using **File → Upload notebook**.
2. Run its first code cell. If the package is not already in the runtime, upload the same code ZIP when prompted. This is a small code package, not the 5 GB image database.
3. Mount your own Google Drive and edit `BASE_DIR`. The default example is `/content/drive/MyDrive/AMD_OCT`.
4. Place the already anonymized images under `BASE_DIR/data/Normal`, `iORA`, `cORA`, `iRORA`, and `cRORA`. Alternatively, place a verified feature NPZ under `BASE_DIR/features` and start at notebook 03.
5. Follow the workflow below. Results are written to a fresh timestamped directory in your own Drive.

The source folders are not publication destinations: all code, data and documentation can be placed together in **one Zenodo record**. No GitHub account or second repository is required. This package performs no publication or remote upload.

## Workflow

| Notebook | Purpose | Input |
|---|---|---|
| `notebooks/00_dataset_inventory.ipynb` | Check category counts and native image dimensions | Anonymized PNG folders |
| `notebooks/01_train_custom_cnn.ipynb` | Train a new five-block CNN with Dense(256) features | Anonymized PNG folders; GPU recommended |
| `notebooks/02_extract_features.ipynb` | Extract 256 custom CNN or 512 VGG16 features | PNG folders and a trained custom model, or frozen ImageNet VGG16 |
| `notebooks/03_compare_cached_features.ipynb` | Configure classifier, extractor and repetitions | Feature NPZ |
| `notebooks/04_inspect_reported_results.ipynb` | Inspect the article's archived 90 evaluations | Included aggregate CSV |
| `notebooks/principal/` | Individual entry points, retaining the supplied classifier notebook filenames | Custom CNN or VGG16 caches |
| `notebooks/exploratory/` | Additional ResNet50/EfficientNet 3-NN entry points | Verified exploratory caches |

The principal paths include eight custom CNN classifiers and seven VGG16 classifiers. The supplied source archive did not contain a standalone implementation of the **VGG16 voting ensemble**. Its five reported scores are preserved in the archived results, but no new implementation is invented or presented as the original. Consequently, 15 principal and two exploratory configurations are executable from verified supplied classifier definitions.

## Scientific scope

- Institutional dataset described in the article: **7,321 images from 2,171 patients**, native size **938 × 625 pixels**.
- Class counts confirmed by the authors: Normal 4,406; iORA 603; cORA 683; iRORA 653; cRORA 976.
- Inputs: 384 × 384, three channels; historical cache-extraction intensity rescaling by 1/255.
- Custom CNN: Flatten 12,800 values followed by **256 dense activations** for downstream classifiers. VGG16: **512 features**.
- Repeated 80:20 **image-level** holdouts over cached representations, without stratification or patient grouping. The extractor is not relearned inside those holdouts.
- Five repetitions and unseeded partitions remain the default. Saved partition indices make each new run traceable. Setting `SPLIT_SEED` creates a new controlled reanalysis; it does not reconstruct the unpublished historical splits.
- Model definitions preserve the classifier-specific settings in the supplied notebooks, including differences between custom CNN and VGG16.
- Reported results and newly generated outputs are kept separate. New runs are not expected to reproduce rounded article values exactly.

These are internal image-level analyses, not patient-disjoint or external clinical evaluations. A future patient-level analysis requires verified anonymous patient identifiers. Sequential image filenames alone cannot recover patient grouping.

## Files and provenance

- `src/amd_oct`: shared data validation, feature extraction, classifier definitions, metrics and saved run outputs.
- `docs/PROVENANCE.md`: source selection and portability changes.
- `docs/source_provenance.json`: original source notebook paths, hashes and relevant classifier cells.
- `docs/source_inventory.csv`: inventory of all 48 supplied notebooks, including one invalid JSON file excluded from executable entry points.
- `results/reported`: aggregate CSV and anatomical confusion counts from the manuscript; no individual OCT images.
- `tests/test_core.py`: label-order, data-validation, metric and output tests.
- `docs/VALIDATION.md`: checks performed for this release.

There are **no original image files, feature caches, fitted clinical models, notebook outputs, Drive sharing links or account credentials** in this code ZIP. Add only the approved anonymized dataset to the same deposit. The supplied Dense32 model is a development variant and is deliberately not packaged as the study's Dense256 extractor.

## Dependencies and local execution

Install `requirements.txt`. Install `requirements-neural.txt` for TensorFlow-dependent paths. Version ranges are portability constraints, **not a claim about the original Colab environment**. Every new experiment records installed versions in `run_config.json`.

For local use, add the `src` directory to `PYTHONPATH` and import `amd_oct`. To run checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Citation and reuse

`CITATION.cff` contains author and version metadata. Add the DOI assigned to the actual repository record after publication; do not substitute the paper DOI for the dataset/code DOI. Consult `LICENSE_NOTICE.md` for the current licensing status. The institution and authors should select the data and code reuse terms for the deposit.
