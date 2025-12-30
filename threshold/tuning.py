from sklearn.metrics import f1_score, classification_report
import numpy as np

def find_best_threshold_f1(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold

def find_best_threshold_recall_precision(y_true, y_prob,
                        min_recall=0.70,
                        min_precision=0.40):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        report = classification_report(
            y_true, y_pred, output_dict=True
        )
        recall = report['1']['recall']
        precision = report['1']['precision']

        if recall >= min_recall and precision >= min_precision:
            best_threshold = t
            break

    return best_threshold