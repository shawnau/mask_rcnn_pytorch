import unittest

from configuration import Configuration
from torch.utils.data import DataLoader
from loader.sampler import *

from loader.coco.dataset import CocoDataset
from loader.coco.dataset import train_augment, valid_augment, train_collate


class TestCocoLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        # loader
        train_dataset = CocoDataset(self.cfg, self.cfg.data_dir, dataType='train2017', mode='train', transform=train_augment)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.cfg.batch_size,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_collate)

        valid_dataset = CocoDataset(self.cfg, self.cfg.data_dir, dataType='val2017', mode='train', transform=valid_augment)
        self.valid_loader = DataLoader(
            valid_dataset,
            sampler=FixLengthRandomSampler(valid_dataset, length=self.cfg.batch_size),
            batch_size=self.cfg.batch_size,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_collate)

    def test_load(self):
        """
        inputs: torch tensor (B, C, H, W)
        truth_boxes: [ndarray]: list of B (N, 4) ndarray
        truth_labels: [ndarray]: list of B (N, ) ndarray
        truth_instances: [ndarray]: list of B (H, W) ndarray
        indices: [int]: list of int, indices of the image
        """
        print('test training set')
        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            print(indices)

        for i in range(20):
            print('test valid set %d'%i)
            for inputs, truth_boxes, truth_labels, truth_instances, indices in self.valid_loader:
                print(indices)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests([
        TestCocoLoader('test_load')
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)