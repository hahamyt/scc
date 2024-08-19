import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.utils import misc

class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label.type_as(pred) - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True).clamp(0, 60000//2)     # Clamp For FP16 training NAN issue (5*10-8, 65504)
        mult = sw_sum / (beta_sum + self._eps)
        if torch.isinf(mult).sum():                 # 防止半精度数值溢出
            mult = (sw_sum.type(torch.float32) / (beta_sum.type(torch.float32) + self._eps)).clamp(0, 60000//3)
            mult = mult.type_as(pred)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        if ((pt + self._eps)==0).sum()>0:
            self._eps = 1e-6
        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=pred.dtype).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
        if any(loss.isnan()):
            loss = (pred - pred).mean()
        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)

# class NormalizedFocalLossSigmoid(nn.Module):
#     def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
#                  from_sigmoid=False, detach_delimeter=True,
#                  batch_axis=0, weight=None, size_average=True,
#                  ignore_label=-1):
#         super(NormalizedFocalLossSigmoid, self).__init__()
#         self._axis = axis
#         self._alpha = alpha
#         self._gamma = gamma
#         self._ignore_label = ignore_label
#         self._weight = weight if weight is not None else 1.0
#         self._batch_axis = batch_axis
#
#         self._from_logits = from_sigmoid
#         self._eps = eps
#         self._size_average = size_average
#         self._detach_delimeter = detach_delimeter
#         self._max_mult = max_mult
#         self._k_sum = 0
#         self._m_max = 0
#
#     def forward(self, pred, label):
#         one_hot = label > 0.5
#         sample_weight = label != self._ignore_label
#
#         if not self._from_logits:
#             pred = torch.sigmoid(pred)
#
#         alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
#         pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))
#
#         beta = (1 - pt) ** self._gamma
#
#         sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
#         beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
#         mult = sw_sum / (beta_sum + self._eps)
#         if self._detach_delimeter:
#             mult = mult.detach()
#         beta = beta * mult
#         if self._max_mult > 0:
#             beta = torch.clamp_max(beta, self._max_mult)
#
#         with torch.no_grad():
#             ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
#             sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
#             if np.any(ignore_area == 0):
#                 self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()
#
#                 beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
#                 beta_pmax = beta_pmax.mean().item()
#                 self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax
#
#         loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
#         loss = self._weight * (loss * sample_weight)
#
#         if self._size_average:
#             bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
#             loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
#         else:
#             loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
#
#         return loss
#
#     def log_states(self, sw, name, global_step):
#         sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
#         sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class BinaryDiceLoss(nn.Module):
    """ Dice Loss for binary segmentation
    """

    def forward(self, pred, label):
        batchsize = pred.size(0)

        # convert probability to binary label using maximum probability
        input_pred, input_label = pred.max(1)
        input_pred *= input_label.float()

        # convert to floats
        input_pred = input_pred.float()
        target_label = label.float()

        # convert to 1D
        input_pred = input_pred.view(batchsize, -1)
        target_label = target_label.view(batchsize, -1)

        # compute dice score
        intersect = torch.sum(input_pred * target_label, 1)
        input_area = torch.sum(input_pred * input_pred, 1)
        target_area = torch.sum(target_label * target_label, 1)

        sum = input_area + target_area
        epsilon = torch.tensor(1e-6)

        # batch dice loss and ignore dice loss where target area = 0
        batch_loss = torch.tensor(1.0) - (torch.tensor(2.0) * intersect + epsilon) / (sum + epsilon)
        loss = batch_loss.mean()

        return loss

def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()
        self.critiation = weighted_focal_l1_loss

    def forward(self, pred, gt):
        B = gt.shape[0]
        # gt = F.interpolate(gt, scale_factor=0.07142857142857142).view(B, -1)
        gt = F.interpolate(gt, scale_factor=0.0625).view(B, -1)
        loss = 0.0
        for i in range(len(pred)):
            pos_sim = pred[i]['pos_sim']
            loss = loss + self.critiation(pos_sim, gt)
            if pred[i]['neg_sim'] is not None:
                neg_sim = pred[i]['neg_sim']
                loss = loss + self.critiation(neg_sim, 1 - gt)


        return loss/len(pred)


class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()
        # self.critiation1 = nn.MSELoss(reduction='none')
        # self.critiation2 = nn.MSELoss(reduction='none')
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2, eps=1e-7)

    def forward(self, pred, prev, gt):
        B = gt.shape[0]
        anchor = gt.view(B, -1)
        positive = pred.view(B, -1)
        negative = prev.view(B, -1)
        loss = self.triplet_loss(anchor, positive, negative)

        # current_loss = self.critiation1(pred, gt)
        # prev_loss = self.critiation2(prev, gt)
        # # 当前损失越大，之前损失越低，说明当前预测结果变差了，此时损失增大;
        # # 反之，当前损失小，之前损失大，说明当前预测结果变好了，此时损失
        # loss = torch.where(current_loss > prev_loss, 1, 0).mean()
        return loss