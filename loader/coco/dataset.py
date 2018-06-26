import os
import cv2
import torch
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
import numpy as np
from skimage.transform import resize, rescale


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
        self.cls_names = [cat['name'] for cat in self.coco.loadCats(cls_ids)]
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


def resize_image(image, min_dim=None, max_dim=None, padding=True):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), mode='constant')
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = rescale(mask, scale, mode='constant')
    mask = np.pad(mask, padding[:2], mode='constant', constant_values=0)
    return mask


def train_augment(image, instances, labels, index):
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    image, window, scale, padding = resize_image(
        image,
        min_dim=IMAGE_MIN_DIM,
        max_dim=IMAGE_MAX_DIM,
        padding=True)

    ret_instances = []
    boxes = []
    for instance in instances:
        instance = instance.astype(np.float32)
        instance = resize_mask(instance, scale, padding)
        ret_instances.append(instance)
        boxes.append(instance_to_box(instance))

    # image read from opencv has the dimension of (Height, Width, Channels).
    # image resized from skimage will div 255 automatically
    input_image = torch.from_numpy(image.transpose((2, 0, 1))).float()

    assert len(instances) == len(labels)
    labels = np.array(labels)
    ret_instances = np.array(ret_instances).astype(np.float32)
    boxes = np.array(boxes).astype(np.float32)
    return input_image, boxes, labels, ret_instances, index


def valid_augment(image, instances, labels, index):
    return train_augment(image, instances, labels, index)


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