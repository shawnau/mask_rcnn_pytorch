import torch
import cv2
import numpy as np

from net.layer.mask.mask_utils import instance_to_binary
from net.layer.rpn.rpn_utils import rpn_decode
from net.layer.rcnn.rcnn_utils import rcnn_decode
from net.utils.box_utils import clip_boxes, filter_boxes
from net.utils.func_utils import np_softmax, np_sigmoid, to_tensor

if torch.cuda.is_available():
    from net.lib.gpu_nms.gpu_nms import gpu_nms as nms_func
else:
    from net.lib.cython_nms.cython_nms import cython_nms as nms_func

from net.lib.box_overlap.cython_box_overlap import cython_box_overlap


def _nms(cfg, mode, head, decode, images, logits, deltas, anchor_boxes=None, rpn_proposals=None):
    """
    used for rpn and rcnn nms_func
    This function:
    1. Do non-maximum suppression on given window and logistic score
    2. filter small ret_proposals, crop border
    3. decode bbox regression

    :param cfg: configure
    :param mode: mode. e.g. 'train', 'test', 'eval'
    :param images: a batch of input images
    :param anchor_boxes: all anchor boxes in a batch, list of coords, e.g.
               [[x0, y0, x1, y1], ...], a total of 16*16*3 + 32*32*3 + 64*64*3 + 128*128*3
    :param logits_np: (B, N, 2) NOT nomalized
               [[0.7, 0.5], ...]
    :param deltas_np: (B, N, 2, 4)
               [[[t1, t2, t3, t4], [t1, t2, t3, t4]], ...]
    :return: all proposals in a batch. e.g.
        [i, x0, y0, x1, y1, score, label]
        proposals[0]:   image idx in the batch
        proposals[1:5]: bbox
        proposals[5]:   probability of foreground (background skipped)
        proposals[6]:   class label, 1 fore foreground, 0 for background, here we only return 1
    """
    if mode in ['train']:
        nms_prob_threshold = cfg.rpn_train_nms_pre_score_threshold if head == 'rpn' else cfg.rcnn_train_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_train_nms_overlap_threshold if head == 'rpn' else cfg.rcnn_train_nms_overlap_threshold
        nms_min_size = cfg.rpn_train_nms_min_size if head == 'rpn' else cfg.rcnn_train_nms_min_size

    elif mode in ['valid', 'test', 'eval']:
        nms_prob_threshold = cfg.rpn_test_nms_pre_score_threshold if head == 'rpn' else cfg.rcnn_test_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_test_nms_overlap_threshold if head == 'rpn' else cfg.rcnn_test_nms_overlap_threshold
        nms_min_size = cfg.rpn_test_nms_min_size if head == 'rpn' else cfg.rcnn_test_nms_min_size

        if mode in ['eval']:
            nms_prob_threshold = 0.05  # set low numbe r to make roc curve.
    else:
        raise ValueError('rpn_nms(): invalid mode = %s?' % mode)

    num_classes = 2 if head == 'rpn' else cfg.num_classes
    logits_np = logits.detach().cpu().numpy()
    deltas_np = deltas.detach().cpu().numpy() if head == 'rpn' else deltas.detach().cpu().numpy().reshape(-1, num_classes, 4)
    batch_size, _, height, width = images.size()

    # non-max suppression
    ret_proposals = []
    for img_idx in range(batch_size):
        pic_proposals = [np.empty((0, 7), np.float32)]
        if head == 'rpn':
            assert anchor_boxes is not None
            raw_box = anchor_boxes
            prob_distrib = np_softmax(logits_np[img_idx])  # (N, 2)
            delta_distrib = deltas_np[img_idx]  # (N, 2, 4)
        else:  # rcnn
            rpn_proposals_np = rpn_proposals.detach().cpu().numpy()
            select = np.where(rpn_proposals_np[:, 0] == img_idx)[0]
            if len(select) == 0:
                return torch.zeros((1, 7)).to(cfg.device)
            raw_box = rpn_proposals_np[select, 1:5]
            prob_distrib = np_softmax(logits_np[select])  # <todo>why not use np_sigmoid?
            delta_distrib = deltas_np[select]

        # skip background
        for cls_idx in range(1, num_classes):  # 0 for background, 1 for foreground
            index = np.where(prob_distrib[:, cls_idx] > nms_prob_threshold)[0]
            if len(index) > 0:
                valid_box = raw_box[index]
                prob  = prob_distrib[index, cls_idx].reshape(-1, 1)
                delta = delta_distrib[index, cls_idx]
                # bbox regression, do some clip/filter
                box = decode(valid_box, delta)
                box = clip_boxes(box, width, height)  # take care of borders
                keep = filter_boxes(box, min_size=nms_min_size)  # get rid of small boxes

                if len(keep) > 0:
                    box = box[keep]
                    prob = prob[keep]
                    keep = nms_func(np.hstack((box, prob)), nms_overlap_threshold)

                    proposal = np.zeros((len(keep), 7), np.float32)
                    proposal[:, 0] = img_idx
                    proposal[:, 1:5] = np.around(box[keep], 0)
                    proposal[:, 5] = prob[keep, 0]
                    proposal[:, 6] = cls_idx
                    pic_proposals.append(proposal)

        pic_proposals = np.vstack(pic_proposals)
        ret_proposals.append(pic_proposals)

    ret_proposals = np.vstack(ret_proposals)
    ret_proposals = to_tensor(ret_proposals, cfg.device)
    return ret_proposals


def rpn_nms(cfg, mode, images, anchor_boxes, logits_flat, deltas_flat):
    return _nms(cfg, mode, 'rpn', rpn_decode, images, logits_flat, deltas_flat, anchor_boxes=anchor_boxes)


def rcnn_nms(cfg, mode, images, rpn_proposals, logits, deltas):
    return _nms(cfg, mode, 'rcnn', rcnn_decode, images, logits, deltas, rpn_proposals=rpn_proposals)


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

    proposals   = proposals.detach().cpu().numpy()
    mask_logits = mask_logits.detach().cpu().numpy()
    mask_probs  = np_sigmoid(mask_logits)

    b_multi_masks = []
    b_mask_proposals = []
    b_mask_instances = []
    batch_size, C, H, W = images.size()
    for b in range(batch_size):
        multi_masks = np.zeros((H, W), np.float32)
        mask_proposals = []
        mask_instances = []
        num_keeps = 0

        index = np.where((proposals[:, 0] == b) & (proposals[:, 5] > pre_score_threshold))[0]
        if len(index) > 0:
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

            # compute box overlap, do cython_nms
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

                t = index[k]  # t is the index of box before nms_func
                b, x0, y0, x1, y1, score, label = proposals[t]
                mask_proposals.append(np.array([b, x0, y0, x1, y1, score, label], np.float32))

        if num_keeps == 0 or len(index) == 0:
            mask_proposals = np.zeros((0,7  ),np.float32)
            mask_instances = np.zeros((0,H,W),np.float32)
        else:
            mask_proposals = np.vstack(mask_proposals)
            mask_instances = np.vstack(mask_instances)

        b_mask_proposals.append(mask_proposals)
        b_mask_instances.append(mask_instances)
        b_multi_masks.append(multi_masks)

    b_mask_proposals = np.vstack(b_mask_proposals)
    b_mask_proposals = to_tensor(b_mask_proposals, cfg.device)
    return b_multi_masks, b_mask_instances, b_mask_proposals