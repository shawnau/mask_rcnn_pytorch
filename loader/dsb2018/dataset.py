import os
from os import listdir
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset


class ScienceDataset(Dataset):
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
    def __init__(self, cfg, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()

        self.cfg = cfg
        self.transform = transform
        self.mode = mode

        # read split
        self.ids = [x.split('.')[0] for x in listdir(os.path.join(self.cfg.data_dir, 'images'))]

    def __getitem__(self, index):
        name = self.ids[index]
        image_path = os.path.join(self.cfg.data_dir, 'images', '%s.png' % name)
        image  = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask_path = os.path.join(self.cfg.data_dir, 'multi_masks', '%s.npy' % name)
            multi_mask = np.load(multi_mask_path).astype(np.int32)

            if self.transform is not None:
                return self.transform(image, multi_mask, index)
            else:
                return image, multi_mask, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)

