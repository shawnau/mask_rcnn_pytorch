import numpy as np
import torch
import cv2
from skimage import morphology

from net.lib.box_overlap.cython_box_overlap import cython_box_overlap
from net.utils.func_utils import np_sigmoid

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


def mask_nms(cfg, images, proposals, mask_logits):
    """
    1. do non-maximum suppression to remove overlapping segmentations
    2. resize the masks from mask head output (28*28) into box size
    3. paste the masks into input image
    :param cfg:
    :param images: (B, C, H, W)
    :param proposals: (B, 7) [i, x0, y0, x1, y1, score, label]
    :param mask_logits: (B, num_classes, 2*crop_size, 2*crop_size)
    :return:
        b_multi_masks: (B, H, W) masks labelled with 1,2,...N (total number of masks)
        b_mask_instances: (B*N, H, W) masks with prob
        b_mask_proposals: (B*N, ) proposals
    """
    overlap_threshold   = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold      = cfg.mask_test_mask_threshold
    mask_min_area       = cfg.mask_test_mask_min_area

    proposals   = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs  = np_sigmoid(mask_logits)

    b_multi_masks = []
    b_mask_proposals = []
    b_mask_instances = []
    batch_size, C, H, W = images.size()
    for b in range(batch_size):
        multi_masks = np.zeros((H, W), np.float32)  # multi masks for a image
        mask_proposals = []  # proposals for a image
        mask_instances = []  # instances for a image
        num_keeps = 0

        index = np.where((proposals[:, 0] == b) & (proposals[:, 5] > pre_score_threshold))[0]
        if len(index) != 0:
            instances = []    # all instances
            boxes = []        # all boxes
            for i in index:
                mask = np.zeros((H, W), np.float32)

                x0, y0, x1, y1 = proposals[i, 1:5].astype(np.int32)
                h, w = y1-y0+1, x1-x0+1
                label = int(proposals[i, 6])    # get label of the instance
                crop = mask_probs[i, label]     # get mask channel of the label
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                # crop = crop > mask_threshold  # turn prob feature map into 0/1 mask
                mask[y0:y1+1, x0:x1+1] = crop   # paste mask into empty mask

                instances.append(mask)
                boxes.append([x0, y0, x1, y1])

            # compute box overlap, do nms
            L = len(index)
            binary = [instance_to_binary(m, mask_threshold, mask_min_area) for m in instances]
            boxes = np.array(boxes, np.float32)
            box_overlap = cython_box_overlap(boxes, boxes)
            instance_overlap = np.zeros((L, L), np.float32)

            # calculate instance overlapping iou
            for i in range(L):
                instance_overlap[i, i] = 1
                for j in range(i+1, L):
                    if box_overlap[i, j] < 0.01:
                        continue

                    x0 = int(min(boxes[i, 0], boxes[j, 0]))
                    y0 = int(min(boxes[i, 1], boxes[j, 1]))
                    x1 = int(max(boxes[i, 2], boxes[j, 2]))
                    y1 = int(max(boxes[i, 3], boxes[j, 3]))

                    mi = binary[i][y0:y1, x0:x1]
                    mj = binary[j][y0:y1, x0:x1]

                    intersection = (mi & mj).sum()
                    union = (mi | mj).sum()
                    instance_overlap[i, j] = intersection/(union + 1e-12)
                    instance_overlap[j, i] = instance_overlap[i, j]

            # non-max-suppression to remove overlapping segmentation
            score = proposals[index, 5]
            sort_idx = list(np.argsort(-score))

            # https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(sort_idx) > 0:
                i = sort_idx[0]
                keep.append(i)
                delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
                sort_idx = [e for e in sort_idx if e not in delete_index]
            # filter instances & proposals
            num_keeps = len(keep)
            for i in range(num_keeps):
                k = keep[i]
                multi_masks[np.where(binary[k])] = i + 1
                mask_instances.append(instances[k].reshape(1, H, W))

                t = index[k]  # t is the index of box before nms
                b, x0, y0, x1, y1, score, label = proposals[t]
                mask_proposals.append(np.array([b, x0, y0, x1, y1, score, label], np.float32))

        if num_keeps==0:
            mask_proposals = np.zeros((0,8  ),np.float32)
            mask_instances = np.zeros((0,H,W),np.float32)
        else:
            mask_proposals = np.vstack(mask_proposals)
            mask_instances = np.vstack(mask_instances)

        b_mask_proposals.append(mask_proposals)
        b_mask_instances.append(mask_instances)
        b_multi_masks.append(multi_masks)

    b_mask_proposals = torch.from_numpy(np.vstack(b_mask_proposals)).to(cfg.device)
    return b_multi_masks, b_mask_instances, b_mask_proposals