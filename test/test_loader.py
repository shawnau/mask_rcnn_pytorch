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
from net.utils.draw import image_show, draw_boxes, instances_to_contour_overlay, instances_to_color_overlay


def visualize_sample(inputs, truth_boxes, truth_labels, truth_instances, indices):
    """
    hardcoded for batch size = 0
    :param inputs:
    :param truth_boxes:
    :param truth_labels:
    :param truth_instances:
    :param indices:
    :return:
    """
    inputs = inputs.numpy()[0]
    truth_boxes = truth_boxes[0]
    truth_labels = truth_labels[0]
    truth_scores = [1.0 for _ in truth_labels]
    truth_instances = truth_instances[0].astype(np.uint8)
    index = indices[0]

    inputs = inputs * 255
    image = inputs.transpose((1, 2, 0))
    image = np.array(image, dtype=np.uint8).copy()  # make contiguous
    # draw boxes and masks
    draw_boxes(image, truth_boxes, truth_labels, truth_scores, color=(0, 0, 255))
    overlay = instances_to_color_overlay(truth_instances)
    cv2.addWeighted(overlay, 0.7, image, 1.0, 0, image)
    return image, index


class TestScienceLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        # loader
        self.train_dataset = ScienceDataset(self.cfg, 'valid', mode='train', transform=dsb_train_augment)
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=dsb_train_collate)

    def test_load(self):
        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            """
            inputs: torch tensor (B, C, H, W)
            truth_boxes: [ndarray]: list of B (N, 4) ndarray
            truth_labels: [ndarray]: list of B (N, ) ndarray
            truth_instances: [ndarray]: list of B (H, W) ndarray
            indices: [int]: list of int, indices of the image
            """
            image, index = visualize_sample(inputs, truth_boxes, truth_labels, truth_instances, indices)

            image_show('%s'%index, image)
            k = cv2.waitKey(0)
            if k == ord(' '):
                continue
            else:
                break


class TestCocoLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        self.cfg.batch_size = 1
        # loader
        self.train_dataset = CocoDataset(self.cfg, self.cfg.data_dir, mode='train', transform=coco_train_augment)
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=coco_train_collate)

    def test_load(self):
        """
        hardcoded for batch size = 0
        inputs: torch tensor (B, C, H, W)
        truth_boxes: [ndarray]: list of B (N, 4) ndarray
        truth_labels: [ndarray]: list of B (N, ) ndarray
        truth_instances: [ndarray]: list of B (H, W) ndarray
        indices: [int]: list of int, indices of the image
        """
        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            truth_labels = [[self.train_dataset.cls_names[int(idx)-1] for idx in truth_labels[0]]]
            image, index = visualize_sample(inputs, truth_boxes, truth_labels, truth_instances, indices)

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