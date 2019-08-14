# from https://github.com/amdegroot/ssd.pytorch/

import numpy as np
import torch


def iou_gt(detect, ground_truths):
    det_size = (detect[2] - detect[0]) * (detect[3] - detect[1])
    detect = detect.resize_(1, 4)
    iou = []
    ioa = []

    for gt in ground_truths:
        gt = gt.resize_(1, 4)
        gt_size = (gt[0][2] - gt[0][0]) * (gt[0][3] - gt[0][1])

        inter_max = torch.max(detect, gt)
        inter_min = torch.min(detect, gt)
        inter_size = max(inter_min[0][2] - inter_max[0][0], 0.) * max(inter_min[0][3] - inter_max[0][1], 0.)

        _iou = inter_size / (det_size + gt_size - inter_size)
        _ioa = inter_size / gt_size

        iou.append(_iou)
        ioa.append(_ioa)
    return iou, ioa


def true_false_positive(detects, ground_truths, label, score, npos, gt_label, iou_threshold=0.5, conf_threshold=0.01):
    '''
    '''
    for det, gt in zip(detects, ground_truths):
        for i, det_c in enumerate(det):
            gt_c = [_gt[:4].data.resize_(1, 4) for _gt in gt if int(_gt[4]) == i]
            iou_c = []
            ioa_c = []
            score_c = []

            for det_c_n in det_c:
                if det_c_n[0] < conf_threshold:
                    break
                if len(gt_c) > 0:
                    _iou, _ioa = iou_gt(det_c_n[1:], gt_c)
                    iou_c.append(_iou)
                    ioa_c.append(_ioa)
                score_c.append(det_c_n[0])

            # No detection 
            if len(iou_c) == 0:
                npos[i] += len(gt_c)
                if len(gt_c) > 0:
                    is_gt_box_detected = np.zeros(len(gt_c), dtype=bool)
                    gt_label[i] += is_gt_box_detected.tolist()
                continue

            labels_c = [0] * len(score_c)

            if len(gt_c) > 0:
                max_overlap_gt_ids = np.argmax(np.array(iou_c), axis=1)
                is_gt_box_detected = np.zeros(len(gt_c), dtype=bool)
                for iters in range(len(labels_c)):
                    gt_id = max_overlap_gt_ids[iters]
                    if iou_c[iters][gt_id] >= iou_threshold:
                        # if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
                        if not is_gt_box_detected[gt_id]:
                            labels_c[iters] = 1
                            is_gt_box_detected[gt_id] = True

            # append to the global label, score
            npos[i] += len(gt_c)
            label[i] += labels_c
            score[i] += score_c
            gt_label[i] += is_gt_box_detected.tolist()

    return label, score, npos, gt_label


def cal_size(detects, ground_truths, size):
    for det, gt in zip(detects, ground_truths):
        for i, det_c in enumerate(det):
            gt_c = [_gt[:4].data.resize_(1, 4) for _gt in gt if int(_gt[4]) == i]
            if len(gt_c) == 0:
                continue
            gt_size_c = [[(_gt[0][2] - _gt[0][0]), (_gt[0][3] - _gt[0][1])] for _gt in gt_c]
            size[i] += gt_size_c
    return size


def precision(_label, _score, _npos):
    recall = []
    precision = []
    ap = []
    for labels, scores, npos in zip(_label[1:], _score[1:], _npos[1:]):
        sorted_indices = np.argsort(scores)
        sorted_indices = sorted_indices[::-1]
        labels = np.array(labels).astype(int)
        true_positive_labels = labels[sorted_indices]
        false_positive_labels = 1 - true_positive_labels
        tp = np.cumsum(true_positive_labels)
        fp = np.cumsum(false_positive_labels)

        rec = tp.astype(float) / float(npos)
        prec = tp.astype(float) / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap += [compute_average_precision(prec, rec)]
        recall += [rec]
        precision += [prec]
    mAP = np.nanmean(ap)
    return precision, recall, mAP


def compute_average_precision(precision, recall):
    """Compute Average Precision according to the definition in VOCdevkit.

    Precision is modified to ensure that it does not decrease as recall
    decrease.

    Args:
      precision: A float [N, 1] numpy array of precisions
      recall: A float [N, 1] numpy array of recalls

    Raises:
      ValueError: if the input is not of the correct format

    Returns:
      average_precison: The area under the precision recall curve. NaN if
        precision and recall are None.

    """
    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(recall,
                                                               np.ndarray):
        raise ValueError("precision and recall must be numpy array")
    if precision.dtype != np.float or recall.dtype != np.float:
        raise ValueError("input must be float numpy array.")
    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision
