import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.arange(0.05, 0.96, 0.01):
        preds = (y_prob >= thr).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = thr

    return best_thr, best_f1