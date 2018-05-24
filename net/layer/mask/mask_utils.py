import numpy as np
import torch
from skimage import morphology
from net.utils.func_utils import binary_cross_entropy_with_logits


def make_empty_masks(cfg, mode, inputs):
    masks = []
    batch_size, C, H, W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks


def instance_to_binary(instance, threshold, min_area):
    binary = instance > threshold
    label  = morphology.label(binary)
    num_labels = label.max()
    if num_labels>0:
        areas    = [(label==c+1).sum() for c in range(num_labels)]
        max_area = max(areas)

        for c in range(num_labels):
            if areas[c] != max_area:
                binary[label==c+1]=0
            else:
                if max_area<min_area:
                    binary[label==c+1]=0
    return binary


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
    select = torch.zeros((batch_size, num_classes)).to(logits.device)
    select.scatter_(1, labels.view(-1, 1), 1)
    select[:, 0] = 0
    select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, dim)).contiguous().byte()

    logits_flat = logits_flat[select].view(-1)
    labels_flat = instances.view(-1)

    return binary_cross_entropy_with_logits(logits_flat, labels_flat)