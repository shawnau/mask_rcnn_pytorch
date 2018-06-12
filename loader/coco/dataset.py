import os
import cv2
import torch
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO

import numpy as np


class CocoDataset(Dataset):
    def __init__(self, cfg, dataDir, dataType='train2017', transform=None, mode='train'):
        """
        :param cfg:
        :param dataDir: root dir for coco dataset
        :param dataType: train2017 or valid2017 for training and validation
        :param transform: train/test time augmentation
        :param mode: train/valid/test
        """
        super(CocoDataset, self).__init__()

        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)

        self.cfg = cfg
        self.transform = transform
        self.mode = mode
        self.coco = COCO(self.annFile)

        # class id re-hashing
        cls_ids = self.coco.getCatIds()
        self.ids_lookup = {}
        for i, cls_id in enumerate(cls_ids):
            self.ids_lookup[cls_id] = i + 1

        # index lookup
        key_list = list(self.coco.imgs.keys())
        self.idx_lookup = {i: key_list[i] for i in range(len(key_list))}

    def __getitem__(self, index):
        coco_index = self.idx_lookup[index]
        imgIds = self.coco.getImgIds(imgIds=[coco_index])
        img_obj = self.coco.loadImgs(imgIds[0])[0]

        image_path = os.path.join(self.dataDir, 'images', self.dataType, img_obj['file_name'])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            instances, labels = self.ann_to_instances(img_obj)

            if self.transform is not None:
                return self.transform(image, instances, labels, index)
            else:
                return image, instances, labels, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.coco.imgs)

    def ann_to_instances(self, img_obj):
        """
        https://zhuanlan.zhihu.com/p/29393415
        :param img_obj: coco image object
        :return:
            multi_mask: (H, W) ndarray multi masks
            ret_bbox: (N, 4) ndarray bboxes
            idx_to_cls: dict to map idx into class
        """
        # H, W = img_obj['height'], img_obj['width']
        annIds = self.coco.getAnnIds(imgIds=img_obj['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        instances = []
        labels = []
        for i, ann in enumerate(anns):
            # bbox = ann['bbox']
            # x0, y0 = bbox[0], bbox[1] - bbox[3]
            # x1, y1 = bbox[0] + bbox[2], bbox[1]
            instances.append(self.coco.annToMask(ann))
            labels.append(self.ids_lookup[ann['category_id']])

        return instances, labels


def instance_to_box(instance):
    H, W = instance.shape[:2]

    y, x = np.where(instance)
    if len(x) == 0 or len(y) == 0:
        return [0., 0., 0., 0.]

    y0 = y.min()
    y1 = y.max()
    x0 = x.min()
    x1 = x.max()
    w = (x1 - x0) + 1
    h = (y1 - y0) + 1

    border = max(2, round(0.05 * (w + h) / 2))

    x0 = x0 - border
    x1 = x1 + border
    y0 = y0 - border
    y1 = y1 + border

    # clip
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W - 1, x1)
    y1 = min(H - 1, y1)

    return [x0, y0, x1, y1]


def train_augment(image, instances, labels, index):
    WIDTH, HEIGHT = 512, 512

    img_height, img_width = image.shape[:2]
    if (img_height, img_width) != (HEIGHT, WIDTH):
        image = cv2.resize(image, (WIDTH, HEIGHT))
        ret_instances = []
        boxes = []
        for instance in instances:
            instance = instance.astype(np.float32)
            instance = cv2.resize(instance, (WIDTH, HEIGHT), cv2.INTER_NEAREST)
            instance = instance.astype(np.int32)
            ret_instances.append(instance)
            boxes.append(instance_to_box(instance))
    else:
        ret_instances = instances
        boxes = [instance_to_box(instance) for instance in instances]

    # image read from opencv has the dimension of (Height, Width, Channels)
    input_image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)

    assert len(instances) == len(labels)
    ret_instances = np.array(ret_instances)
    labels = np.array(labels)
    boxes = np.array(boxes).astype(np.float32)
    return input_image, boxes, labels, ret_instances, index


def valid_augment(image, instances, labels, index):
    WIDTH, HEIGHT = 512, 512

    img_height, img_width = image.shape[:2]
    if (img_height, img_width) != (HEIGHT, WIDTH):
        image = cv2.resize(image, (WIDTH, HEIGHT))
        ret_instances = []
        boxes = []
        for instance in instances:
            instance = instance.astype(np.float32)
            instance = cv2.resize(instance, (WIDTH, HEIGHT), cv2.INTER_NEAREST)
            instance = instance.astype(np.int32)
            ret_instances.append(instance)
            boxes.append(instance_to_box(instance))
    else:
        ret_instances = instances
        boxes = [instance_to_box(instance) for instance in instances]

    # image read from opencv has the dimension of (Height, Width, Channels)
    input_image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)

    assert len(instances) == len(labels)
    ret_instances = np.array(ret_instances)
    labels = np.array(labels)
    boxes = np.array(boxes).astype(np.float32)
    return input_image, boxes, labels, ret_instances, index


def train_collate(batch):
    """
    :param batch: a batch of data returned by DataSet
    :return:
        collated batch of data. list
    """
    batch_size = len(batch)

    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    indices   =             [batch[b][4]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, indices]