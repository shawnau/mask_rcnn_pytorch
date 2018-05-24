import torch
import torch.nn.functional as F
from net.configuration import Configuration

cfg = Configuration()

def rcnn_reg_loss(labels, deltas, targets, deltas_sigma=1.0):
    """
    :param labels: (\sum B_i*num_proposals_i, )
    :param deltas: (\sum B_i*num_proposals_i, num_classes*4)
    :param targets:(\sum B_i*num_proposals_i, 4)
    :param deltas_sigma: float
    :return: float
    """
    batch_size, num_classes_mul_4 = deltas.size()
    num_classes = num_classes_mul_4 // 4
    deltas = deltas.view(batch_size, num_classes, 4)

    num_pos_proposals = len(labels.nonzero())
    if num_pos_proposals > 0:
        # one hot encode. select could also seen as mask matrix
        select = torch.zeros((batch_size, num_classes)).to(cfg.device)
        select.scatter_(1, labels.view(-1, 1), 1)
        select[:, 0] = 0  # bg is 0, label starts from 1

        select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, 4)).contiguous().byte()
        deltas = deltas[select].view(-1, 4)

        deltas_sigma2 = deltas_sigma * deltas_sigma
        return F.smooth_l1_loss(deltas * deltas_sigma2, targets * deltas_sigma2,
                                size_average=False) / deltas_sigma2 / num_pos_proposals
    else:
        return torch.FloatTensor(0.0, requires_grad=True).to(cfg.device)


def rcnn_cls_loss(logits, labels):
    """
    :param logits: (\sum B_i*num_proposals_i, num_classes)
    :param labels: (\sum B_i*num_proposals_i, )
    :return: float
    """
    return F.cross_entropy(logits, labels, size_average=True)