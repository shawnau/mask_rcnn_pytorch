import os
import cv2
import torch
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO

from loader.transforms import np, random_crop_transform, fix_crop_transform


class CocoDataset(Dataset):
    """
    train mode:
        :return:
        image: (H, W, C) numpy array
        multi_mask: a map records masks. e.g.
            [[0, 1, 1, 0],
             [2, 0, 0, 3],
             [2, 0, 3, 3]]
            for 3 masks in a 4*4 input
        index: index of the image (unique)
    """
    def __init__(self, cfg, dataDir, dataType='train2017', transform=None, mode='train'):
        super(CocoDataset, self).__init__()

        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)

        self.cfg = cfg
        self.transform = transform
        self.mode = mode
        self.coco = COCO(self.annFile)

    def __getitem__(self, index):
        imgIds = self.coco.getImgIds(imgIds=[index])
        img_obj = self.coco.loadImgs(imgIds[0])[0]

        image_path = os.path.join(self.dataDir, 'images', self.dataType, img_obj['file_name'])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask, idx_to_cls = self.instances_to_multi_mask(img_obj)

            if self.transform is not None:
                return self.transform(image, multi_mask, idx_to_cls, index)
            else:
                return image, multi_mask, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.coco.imgs)

    def instances_to_multi_mask(self, img_obj):
        """
        https://zhuanlan.zhihu.com/p/29393415
        :param img_obj: coco image object
        :return:
            multi_mask: (H, W) ndarray multi masks
            ret_bbox: (N, 4) ndarray bboxes
            idx_to_cls: dict to map idx into class
        """
        H, W = img_obj['height'], img_obj['width']
        annIds = self.coco.getAnnIds(imgIds=img_obj['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        multi_mask = np.zeros((H, W), np.int32)
        idx_to_cls = {}
        for i, ann in enumerate(anns):
            # bbox = ann['bbox']
            # x0, y0 = bbox[0], bbox[1] - bbox[3]
            # x1, y1 = bbox[0] + bbox[2], bbox[1]

            cls = ann['category_id']
            binary = self.coco.annToMask(ann)
            assert binary.shape == multi_mask.shape
            multi_mask[binary == 1] = i + 1
            idx_to_cls[i+1] = cls

        return multi_mask, idx_to_cls


def multi_mask_to_annotation(multi_mask, idx_to_cls):
    """
    :param multi_mask: a map records masks. e.g.
        [[0, 1, 1, 0],
         [2, 0, 0, 3],
         [2, 0, 3, 3]]
        for 3 masks in a 4*4 input
    :return:
        boxes: lists of secondary diagonal coords. e.g.
            [[x0, y0, x1, y1], ...]
        labels: currently all labels are 1 (for foreground only)
        instances: list of one vs all masks. e.g.
            [[[0, 1, 1, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]], ...]
            for thr first mask of all masks, a total of 3 lists in this case
    """
    H,W      = multi_mask.shape[:2]
    boxes      = []
    labels     = []
    instances  = []

    idxs = [x for x in np.unique(multi_mask) if x != 0]
    for idx in idxs:
        mask = (multi_mask == idx)

        y, x = np.where(mask)
        y0 = y.min()
        y1 = y.max()
        x0 = x.min()
        x1 = x.max()
        w = (x1-x0)+1
        h = (y1-y0)+1

        border = max(2, round(0.2*(w+h)/2))

        x0 = x0-border
        x1 = x1+border
        y0 = y0-border
        y1 = y1+border

        # clip
        x0 = max(0,x0)
        y0 = max(0,y0)
        x1 = min(W-1,x1)
        y1 = min(H-1,y1)

        boxes.append([x0, y0, x1, y1])
        labels.append(idx_to_cls[idx])
        instances.append(mask)

    boxes     = np.array(boxes,  np.float32)
    labels    = np.array(labels, np.int64)
    instances = np.array(instances, np.float32)

    if len(boxes) == 0:
        boxes     = np.zeros((0, 4), np.float32)
        labels    = np.zeros((0, ), np.int64)
        instances = np.zeros((0, H, W), np.float32)

    return boxes, labels, instances


def train_augment(image, multi_mask, idx_to_cls, index):
    WIDTH, HEIGHT = 256, 256

    # image, multi_mask = \
    #    random_shift_scale_rotate_transform(
    #        image, multi_mask,
    #        shift_limit=[0, 0],
    #        scale_limit=[1/2, 2],
    #        rotate_limit=[-45, 45],
    #        borderMode=cv2.BORDER_REFLECT_101,
    #        u=0.5)
    #
    image, multi_mask = random_crop_transform(image, multi_mask, WIDTH, HEIGHT, u=0.5)
    # image, multi_mask = random_horizontal_flip_transform(image, multi_mask, 0.5)
    # image, multi_mask = random_vertical_flip_transform(image, multi_mask, 0.5)
    # image, multi_mask = random_rotate90_transform(image, multi_mask, 0.5)

    # image read from opencv has the dimension of (Height, Width, Channels)
    input_image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
    boxes, labels, instances = multi_mask_to_annotation(multi_mask, idx_to_cls)

    return input_image, boxes, labels, instances, index


def valid_augment(image, multi_mask, idx_to_cls, index):
    WIDTH, HEIGHT = 256, 256

    image, multi_mask = fix_crop_transform(image, multi_mask, -1, -1, WIDTH, HEIGHT)
    input_image = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask, idx_to_cls)

    return input_image, box, label, instance, index


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