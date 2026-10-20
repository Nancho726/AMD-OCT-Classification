# Source selection and portability changes

Source: `Códigos - Últimas versiones-20261013T151325Z-1-002.zip`, supplied by the authors on 13 October 2026. Original notebook filenames are retained for the individual classifier entry points. SHA-256 hashes and source model cells are recorded in the adjacent provenance files.

## Preserved experiment definitions

| Configuration | Source settings |
|---|---|
| Custom CNN SVM | RBF, sklearn defaults, probability=False; decision-function ROC |
| VGG16 SVM | RBF, probability=True; probability ROC, as in its supplied notebook |
| 3-NN | Three neighbors for every supplied extractor |
| Random forest | 100 trees; custom CNN estimator seed 42; VGG16 estimator seed None |
| XGBoost | Custom CNN eval_metric=logloss and seed 42; VGG16 eval_metric=mlogloss and seed None |
| GBM | 100 estimators, learning rate 0.1, depth 3; estimator seed None |
| Custom CNN feedforward | Keras Dense 1024/512/256/128; BN after 1024/512/256; dropout 0.5 after first two; Adam 0.001; 30 epochs, batch 32 |
| VGG16 feedforward | sklearn MLP 64/32; max_iter=500 |
| Custom CNN RNN | Input (1,d); SimpleRNN 64 ReLU, dropout .4, Dense32; 30 epochs, batch16 |
| VGG16 RNN | Input (1,d); SimpleRNN 64 ReLU, Dense32; 50 epochs, batch32; no dropout |
| Custom CNN ensemble | Equal soft voting of RBF SVM and MLP512/256, averaged with RNN128 tanh on (d,1); effective weights .25/.25/.50 |

The 16 primary and two additional configurations in the article remain represented by 90 archived aggregate results. Only 17 supplied classifier implementations could be traced; the VGG16 voting ensemble is archived as reported data only. No matching standalone implementation was found in the supplied archive. One unrelated development notebook is invalid JSON and is not repaired by guessing truncated content.

## Changes made for the organized release

- Shared configuration replaces personal hard-coded Drive paths.
- Removed notebook outputs, execution counters, widgets, embedded image previews and Colab identifiers.
- Numeric cache labels are named using verified class order; confusion matrices are reordered on **both axes** for display. ROC score columns retain the model's numerical label order.
- Removed obsolete `use_label_encoder=False` from XGBoost constructors; the original objective and other model parameters are not retuned.
- New runs save partition indices, predictions, raw scores, per-class ROC arrays and environment metadata.
- Feature NPZ files include class names and anonymized relative image paths. Legacy NPZ inputs require an explicitly verified label order.
- Extraction targets Dense256 explicitly rather than relying on fragile negative layer indices. It rejects Dense32 models. The VGG16 frozen base with average pooling returns the same 512-dimensional inference representation as its pooled/dropout feature layer; no fine-tuning is introduced.
- CNN training uses the supplied five-block Dense256 architecture, learning rate, callbacks and mixed precision. Training generator batch size is 32. Validation shuffle=False is a deliberate portability/reporting change to keep validation predictions aligned; it does not reconstruct the original training session.

## Data and reproducibility boundaries

The curated manuscript counts (Normal 4406, iORA 603, cORA 683, iRORA 653, cRORA 976) are preserved as author-confirmed dataset descriptors. The supplied legacy feature caches have 604 iORA and 652 iRORA rows under their verified mapping. Code reports observed input counts rather than changing labels to force agreement. A final anonymized dataset inventory should be retained with the public data.

The original Dense256 weights were not present in the supplied archive; its Dense32 `.keras` file is not an interchangeable substitute. New training and ImageNet downloads produce a new execution, not a claim of bit-identical historical features. Unfixed splits and unrecorded historical environment versions also prevent exact reconstruction of the five historical partitions.

The reported CSV is copied unchanged from the manuscript's supplementary data. No new validation score is substituted for a paper result.

User-facing instructions are in English. Provenance snippets omit original Spanish comments and translate printed messages; recorded notebook hashes continue to identify the unmodified original sources. Original filenames and proper institution/author names are retained.
