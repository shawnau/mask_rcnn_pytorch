import torch
import torch.nn.functional as F
import numpy as np


def to_tensor(source, device):
    if type(source) is np.ndarray:
        return torch.from_numpy(source).to(device)
    elif type(source) is torch.Tensor:
        return source
    else:
        raise TypeError('unknown data type')


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# https://github.com/pytorch/pytorch/issues/563
def weighted_focal_loss_for_cross_entropy(logits, labels, weights, gamma=2.):
    """
    assume N samples, each label has K category
    :param logits: (N, K)  unnormalized foreground/background score
    :param labels: (N, )   {K} for k different category
    :param weights: (N, )  float \in (0,1] for rareness, otherwise 0 (don't care)
    :param gamma: float
    :return: float loss
    """
    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)

    loss = - log_probs * (1 - probs) ** gamma
    loss = (weights * loss).sum()/(weights.sum()+1e-12)

    return loss.sum()


# http://geek.csdn.net/news/detail/126833
def weighted_binary_cross_entropy_with_logits(logits, labels, weights):
    """
    assume N samples
    :param logits: (N, 2)  unnormalized foreground/background score
    :param labels: (N, )   {0, 1}
    :param weights: (N, )  float \in [0,1] for rareness
    :return: float loss
    """
    loss = logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs()))
    loss = (weights*loss).sum()/(weights.sum()+1e-12)
    return loss


def weighted_cross_entropy_with_logits(logits, labels, weights):
    """
    assume N samples, each label has K category
    :param logits: (N, K)  unnormalized foreground/background score
    :param labels: (N, )   {K} for k different category
    :param weights: (N, )  float \in (0,1] for rareness, otherwise 0 (don't care)
    :return: float loss
    """
    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    loss = - log_probs
    loss = (weights * loss).sum()/(weights.sum()+1e-12)
    return loss


# original F1 smooth loss from rcnn
def weighted_smooth_l1(predicts, targets, weights, sigma=3.0):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise

        inside_weights  = 1
        outside_weights = 1/num_examples
    '''

    predicts = predicts.view(-1)
    targets  = targets.view(-1)
    weights  = weights.view(-1)

    sigma2 = sigma * sigma
    diffs  =  predicts-targets
    smooth_l1_signs = torch.abs(diffs) < (1.0 / sigma2)
    smooth_l1_signs = smooth_l1_signs.to(predicts.device, dtype=torch.float32)

    smooth_l1_option1 = 0.5 * diffs * diffs * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    loss = smooth_l1_option1*smooth_l1_signs + smooth_l1_option2*(1-smooth_l1_signs)

    loss = (weights*loss).sum()/(weights.sum()+1e-12)

    return loss


# unitbox loss
# https://github.com/zhimingluo/UnitBox_TF/blob/master/UnitBox.py
# https://github.com/zhimingluo/UnitBox_TF/blob/master/IOULoss.py
#
# https://arxiv.org/abs/1608.01471
#  "UnitBox: An Advanced Object Detection Network"
def weighted_iou_loss(predicts, targets, weights):
    (bx0, by0, bx1, by1) = torch.split(predicts,  1, 1)
    (tx0, ty0, tx1, ty1) = torch.split(targets,   1, 1)

    # compute areas
    b = (bx1+bx0 + 1)*(by1+by0 + 1)
    t = (tx1+tx0 + 1)*(ty1+ty0 + 1)

    # compute iou
    ih = (torch.min(by1, ty1) + torch.min(by0, ty0)+ 1)
    iw = (torch.min(bx1, tx1) + torch.min(bx0, tx0)+ 1)
    intersect =  ih*iw
    union = b + t - intersect
    iou   = intersect/(union + 1e-12)

    #loss =
    loss = - torch.log(iou).clamp(max=0)
    loss =  (weights*loss).sum()/(weights.sum()+1e-12)

    return loss


def weighted_l2s(predicts, targets, weights):
    loss =  0.5*(predicts-targets)**2
    loss =  (weights*loss).sum()/(weights.sum()+1e-12)
    return loss

def binary_cross_entropy_with_logits(logits, labels):
    loss = logits.clamp(min=0) - logits * labels + torch.log(1 + torch.exp(-logits.abs()))
    loss = loss.sum() / len(loss)
    return loss