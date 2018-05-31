import os
import cv2
import numpy as np
from glob import glob


class SourceFolder:
    def __init__(self, folder_name):
        self.folder_name = folder_name

    def get_image(self, img_id, flags=cv2.IMREAD_COLOR):
        img_folder = os.path.join(self.folder_name, img_id, 'images')
        img_files = glob(os.path.join(img_folder, '*.png'))
        if len(img_files) == 0:
            img_files = glob(os.path.join(img_folder, '*.tif'))
        assert len(img_files) == 1
        img_file = img_files[0]

        return cv2.imread(img_file, flags)

    def get_masks(self, img_id):
        img = self.get_image(img_id)
        H, W, C = img.shape
        multi_mask = np.zeros((H, W), np.int32)

        mask_folder = os.path.join(self.folder_name, img_id, 'masks')
        mask_files = glob(os.path.join(mask_folder, '*.png'))
        mask_files.sort()

        for j in range(len(mask_files)):
            mask_file = mask_files[j]
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mh, mw = mask.shape
            assert (mh == H) and (mw == W)
            multi_mask[np.where(mask > 128)] = j+1

        return multi_mask


class DataFolder:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.image_folder = os.path.join(self.folder_name, 'images')
        self.mask_folder = os.path.join(self.folder_name, 'multi_masks')

        os.makedirs(self.folder_name, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)

    def get_image(self, img_id, flags=cv2.IMREAD_COLOR):
        return cv2.imread(os.path.join(self.image_folder, '%s.png')%img_id, flags)

    def get_masks(self, img_id):
        return np.load(os.path.join(self.mask_folder, '%s.npy')%img_id).astype(np.int32)


if __name__ == '__main__':
    ids = [x for x in os.listdir('stage1_train') if '.' not in x]
    s_train = SourceFolder('stage1_train')
    d_train = DataFolder('dsb2018')

    for i, img_id in enumerate(ids):
        image = s_train.get_image(img_id)
        multi_masks = s_train.get_masks(img_id)

        cv2.imwrite(os.path.join(d_train.image_folder, '%s.png' % img_id), image)
        np.save(os.path.join(d_train.mask_folder, '%s.npy' % img_id), multi_masks)
        print('%s annotate: %s'%(i, img_id[:5]), end='\r')

    print('Done!')
