import numpy as np


# https://www.kaggle.com/wcukierski/example-metric-implementation
def compute_precision(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects  # TODO is this the definition of true positive?
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def compute_average_precision_for_mask(predict, truth, t_range=np.arange(0.5, 1.0, 0.05)):
    num_truth = len(np.unique(truth))
    num_predict = len(np.unique(predict))

    # Compute intersection between all objects
    intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth, num_predict))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(truth, bins=num_truth)[0]
    area_pred = np.histogram(predict, bins=num_predict)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    precision = []
    average_precision = 0
    for t in t_range:
        tp, fp, fn = compute_precision(t, iou)
        p = tp / (tp + fp + fn)
        precision.append((t, p, tp, fp, fn))
        average_precision += p

    average_precision /= len(precision)
    return average_precision, precision
