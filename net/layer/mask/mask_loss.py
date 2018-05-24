import torch
from net.configuration import Configuration
from net.utils.func_utils import binary_cross_entropy_with_logits

cfg = Configuration()


def mask_loss(logits, labels, instances):
    """
    :param logits:  (\sum B_i*num_proposals_i, num_classes, 2*crop_size, 2*crop_size)
    :param labels:  (\sum B_i*num_proposals_i, )
    :param instances: (\sum B_i*num_proposals_i, 2*crop_size, 2*crop_size)
    :return:
    """
    batch_size, num_classes = logits.size(0), logits.size(1)

    logits_flat = logits.view(batch_size, num_classes, -1)
    dim = logits_flat.size(2)

    # one hot encode
    select = torch.zeros((batch_size, num_classes)).to(cfg.device)
    select.scatter_(1, labels.view(-1, 1), 1)
    select[:, 0] = 0
    select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, dim)).contiguous().byte()

    logits_flat = logits_flat[select].view(-1)
    labels_flat = instances.view(-1)

    return binary_cross_entropy_with_logits(logits_flat, labels_flat)