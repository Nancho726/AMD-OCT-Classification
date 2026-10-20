"""Anatomically named metrics; score columns always follow numeric label order."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc)
from .data import DISPLAY_CLASS_ORDER

def evaluate(y_true, y_pred, scores, class_names):
    y_true, y_pred, scores = np.asarray(y_true), np.asarray(y_pred), np.asarray(scores)
    if scores.shape != (len(y_true), 5):
        raise ValueError('Scores must have shape (number of validation images, 5).')
    order = [class_names.index(c) for c in DISPLAY_CLASS_ORDER]
    cm = confusion_matrix(y_true, y_pred, labels=order)
    reports = classification_report(y_true, y_pred, labels=order,
        target_names=DISPLAY_CLASS_ORDER, output_dict=True, zero_division=0)
    roc, aucs = {}, []
    for i, name in enumerate(class_names):
        binary = (y_true == i).astype(int)
        if len(np.unique(binary)) < 2:
            roc[name] = {'auc': None, 'fpr': [], 'tpr': []}
        else:
            fpr, tpr, _ = roc_curve(binary, scores[:, i])
            value = float(auc(fpr, tpr)); aucs.append(value)
            roc[name] = {'auc': value, 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    specificity = np.divide(tn, tn + fp, out=np.zeros(5, dtype=float), where=(tn + fp) != 0)
    normal = class_names.index('Normal')
    atrophy_true, atrophy_pred = y_true != normal, y_pred != normal
    subtype = float(np.mean(y_true[atrophy_true] == y_pred[atrophy_true])) if atrophy_true.any() else None
    metrics = {'accuracy': float(accuracy_score(y_true, y_pred)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'macro_specificity': float(specificity.mean()),
        'macro_auc': float(np.mean(aucs)) if len(aucs) == 5 else None,
        'exact_subtype_among_atrophy': subtype,
        'atrophy_sensitivity': float(atrophy_pred[atrophy_true].mean()) if atrophy_true.any() else None,
        'normal_specificity': float((~atrophy_pred[~atrophy_true]).mean()) if (~atrophy_true).any() else None}
    return metrics, reports, cm, roc

def save_figures(folder, cm, roc):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from pathlib import Path
    folder = Path(folder)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=DISPLAY_CLASS_ORDER).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set(xlabel='Predicted category', ylabel='Reference category')
    fig.tight_layout(); fig.savefig(folder / 'confusion_matrix.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in DISPLAY_CLASS_ORDER:
        item = roc[name]
        if item['auc'] is not None:
            ax.plot(item['fpr'], item['tpr'], label=f"{name} (AUC={item['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--'); ax.legend(loc='lower right')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='One-versus-rest ROC')
    fig.tight_layout(); fig.savefig(folder / 'roc.png', dpi=180); plt.close(fig)
