import torch
from loader.transforms import *
from torch.utils.data.sampler import *
from loader.dsb2018.dataset import ScienceDataset


def multi_mask_to_annotation(multi_mask):
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
            todo: use class label for different classes
    """
    H,W      = multi_mask.shape[:2]
    boxes      = []
    labels    = []
    instances = []

    num_masks = multi_mask.max()
    for i in range(num_masks):
        mask = (multi_mask == (i+1))
        if mask.sum() > 1:

            y, x = np.where(mask)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1

            # border = max(1, round(0.1*min(w,h)))
            # border = 0
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

            boxes.append([x0,y0,x1,y1])
            labels.append(1)  # <todo> support multiclass
            instances.append(mask)

    boxes     = np.array(boxes,  np.float32)
    labels    = np.array(labels, np.int64)
    instances = np.array(instances, np.float32)

    if len(boxes)==0:
        boxes     = np.zeros((0, 4), np.float32)
        labels    = np.zeros((0, ), np.int64)
        instances = np.zeros((0, H, W), np.float32)

    return boxes, labels, instances


def train_augment(image, multi_mask, index):
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
    boxes, labels, instances = multi_mask_to_annotation(multi_mask)

    return input_image, boxes, labels, instances, index


def make_collate(batch):
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


class ConstantSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        # print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        # print ('\tcalling Sampler:__len__')
        return self.num_samples


# see trorch/visualize_utils/data/sampler.py
class FixLengthRandomSampler(Sampler):
    def __init__(self, data, length=None):
        self.len_data = len(data)
        self.length   = length or self.len_data

    def __iter__(self):
        # print ('\tcalling Sampler:__iter__')

        l=[]
        while 1:
            ll = list(range(self.len_data))
            random.shuffle(ll)
            l = l + ll
            if len(l)>=self.length: break

        l= l[:self.length]
        return iter(l)

    def __len__(self):
        # print ('\tcalling Sampler:__len__')
        return self.length

