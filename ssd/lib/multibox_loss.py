#based on https://github.com/amdegroot/ssd.pytorch/

import torch
import torch.nn.functional as F

from ssd.lib.box_utils import match, log_sum_exp


class MultiBoxLoss(torch.nn.Module):

    def __init__(self, priors, classes):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = classes
        self.negpos_ratio = 3
        self.threshold = 0.5
        self.variance = [0.1, 0.2]
        self.priors = priors

    def forward(self, predictions, targets):
        loc, conf = predictions
        priors = self.priors

        num_priors = priors.shape[0]
        batch_size = num = loc.size(0)

        loc_t = torch.Tensor(batch_size, num_priors, 4).cuda()
        conf_t = torch.LongTensor(batch_size, num_priors).cuda()

        for batch in range(batch_size):
            t = targets[batch][:, :-1].data
            l = targets[batch][:, -1].data
            defaults = priors.data
            if len(t) > 0:
                match(self.threshold, t, defaults, self.variance, l, loc_t, conf_t, batch)

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc)
        loc_p = loc[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf)
        neg_idx = neg.unsqueeze(2).expand_as(conf)
        conf_p = conf[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]

        if len(targets_weighted) == 0:
            loss_c = torch.FloatTensor(torch.zeros([1])).cuda()
        else:
            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
