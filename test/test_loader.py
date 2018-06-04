import unittest
import cv2
import numpy as np
from configuration import Configuration
from torch.utils.data import DataLoader
from loader.dsb2018.dataset import ScienceDataset
from loader.dsb2018.dataset import train_augment as dsb_train_augment
from loader.dsb2018.dataset import train_collate as dsb_train_collate

from loader.coco.dataset import CocoDataset
from loader.coco.dataset import train_augment as coco_train_augment
from loader.coco.dataset import train_collate as coco_train_collate

from loader.sampler import *
from net.utils.draw import image_show, draw_boxes, instances_to_contour_overlay


class TestScienceLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        # loader
        train_dataset = ScienceDataset(self.cfg, 'valid', mode='train', transform=dsb_train_augment)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=dsb_train_collate)

    def test_load(self):
        pass
        # for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
        #     """
        #     inputs: torch tensor (B, C, H, W)
        #     truth_boxes: [ndarray]: list of B (N, 4) ndarray
        #     truth_labels: [ndarray]: list of B (N, ) ndarray
        #     truth_instances: [ndarray]: list of B (H, W) ndarray
        #     indices: [int]: list of int, indices of the image
        #     """
        #     # get batch 0
        #     inputs = inputs.numpy()[0]
        #     truth_boxes = truth_boxes[0]
        #     truth_labels = truth_labels[0]
        #     truth_instances = truth_instances[0].astype(np.uint8)
        #     index = indices[0]
        #     # image
        #     inputs = inputs * 255
        #     image = inputs.transpose((1, 2, 0))
        #     image = np.array(image, dtype=np.uint8).copy() # make contiguous
        #     # draw boxes and masks
        #     draw_boxes(image, truth_boxes, color=(0, 0, 255))
        #     image = instances_to_contour_overlay(truth_instances, image, color=[0, 255, 0])
        #
        #     image_show('%s'%index, image)
        #     k = cv2.waitKey(0)
        #     if k == ord(' '):
        #         continue
        #     else:
        #         break


class TestCocoLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        # loader
        train_dataset = CocoDataset(self.cfg, self.cfg.data_dir, mode='train', transform=coco_train_augment)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=ConstantSampler([9, 25, 30, 34, 36, 42]),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=coco_train_collate)

    def test_load(self):
        """
        inputs: torch tensor (B, C, H, W)
        truth_boxes: [ndarray]: list of B (N, 4) ndarray
        truth_labels: [ndarray]: list of B (N, ) ndarray
        truth_instances: [ndarray]: list of B (H, W) ndarray
        indices: [int]: list of int, indices of the image
        """
        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            # get batch 0
            inputs = inputs.numpy()[0]
            truth_boxes = truth_boxes[0]
            truth_labels = truth_labels[0]
            truth_instances = truth_instances[0].astype(np.uint8)
            index = indices[0]
            # image
            inputs = inputs * 255
            image = inputs.transpose((1, 2, 0))
            image = np.array(image, dtype=np.uint8).copy() # make contiguous
            # draw boxes and masks
            draw_boxes(image, truth_boxes, color=(0, 0, 255))
            image = instances_to_contour_overlay(truth_instances, image, color=[0, 255, 0])

            image_show('%s'%index, image)
            k = cv2.waitKey(0)
            if k == ord(' '):
                continue
            else:
                break


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([
        #TestScienceLoader('test_load'),
        TestCocoLoader('test_load')
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)