import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch

def lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (False), or if it is the last value (True).
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, False
        last = val
    # Report the last value.
    yield last, True


def threshold_search(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)
    F = 2 / (1 / (precision + 1e-18) + 1 / (recall + 1e-18))
    F[F > 1.0] = 0
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    return best_th, best_score

def fbeta_search(y_true, y_proba, beta=0.5):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    F = (1+beta**2) / (1 / (precision + 1e-18) + beta**2 / (recall + 1e-18))

    id = np.argmax(F)
    best_recall = recall[id]
    best_precision = precision[id]

    return best_recall, best_precision, F[id], thresholds[id]

def remove_duplicates(actual, proba, indices, reduce='max'):
    #return max value of tested sample
    assert reduce in ['max', 'min']

    order = True if reduce == 'min' else False
    proba, idx = torch.sort(proba, descending=order)
    actual = actual[idx]
    indices = indices[idx]
    indices, idx = torch.sort(indices, stable=True)
    #indices, idx = torch.sort(indices)
    proba = proba[idx]
    actual = actual[idx]

    # Find first unique index of sample
    #unique, inverse = torch.unique_consecutive(indices, return_inverse=True)
    perm = torch.arange(indices.size(0), dtype=indices.dtype, device=indices.device)

    #Last value will be taked into account only
    perm = indices.new_empty(indices.size(0)).scatter_(0, indices, perm)
    perm = perm[:indices[-1]+1]
    return actual[perm], proba[perm]
