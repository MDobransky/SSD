# from https://github.com/amdegroot/ssd.pytorch/

import torch
from torch.autograd import Function

from ssd.lib.box_utils import decode
from ssd.lib.box_utils import nms


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, priors, classes, conf_threshold=0.2):
        self.num_classes = classes
        self.conf_thresh = conf_threshold
        self.nms_thresh = 0.5
        self.top_k = 100
        self.variance = [0.1, 0.2]
        self.priors = priors

    def forward(self, predictions):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        with torch.no_grad():
            loc, conf = predictions

            batches = loc.size(0)  # batch size
            priors_cnt = self.priors.size(0)

            # if batches == 1:
            # size batch x num_classes x num_priors
            #   conf_preds = conf_data.t().contiguous().unsqueeze(0)
            # else:
            conf_preds = conf.view(batches, priors_cnt, self.num_classes).transpose(2, 1)

            output = torch.zeros(batches, self.num_classes, self.top_k, 5).cuda()

            for i in range(batches):
                decoded_boxes = decode(loc[i], self.priors, self.variance)
                # For each class, perform nms
                conf_scores = conf_preds[i]
                for cl in range(1, self.num_classes):

                    c_mask = conf_scores[cl].gt(self.conf_thresh).nonzero().view(-1)
                    if c_mask.dim() == 0:
                        continue

                    scores = conf_scores[cl][c_mask]
                    boxes = decoded_boxes[c_mask, :]

                    if scores.size(0) == 0:
                        continue

                    ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                    output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                       boxes[ids[:count]]), 1)
            return output
