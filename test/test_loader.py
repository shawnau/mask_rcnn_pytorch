import sys
sys.path.append('../')
import unittest
from torch.utils.data import DataLoader
from loader.dsb2018.train_utils import *
from visualize_utils.draw import image_show, draw_boxes, instances_to_contour_overlay, instances_to_color_overlay


class Configuration:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_dir = 'test_data/'
        self.batch_size = 10


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = Configuration()
        # loader
        train_dataset = ScienceDataset(self.cfg, 'test/data/valid43', mode='train', transform=train_augment)
        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=make_collate)

    def test_load(self):
        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
            """
            inputs: torch tensor (B, C, H, W)
            truth_boxes: [ndarray]: list of B (N, 4) ndarray
            truth_labels: [ndarray]: list of B (N, ) ndarray
            truth_instances: [ndarray]: list of B (H, W) ndarray
            indices: [int]: list of int, indices of the image
            """
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
            image = draw_boxes(image, truth_boxes, color=(0, 0, 255))
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
        TestDataLoader('test_load')
    ])

    runner = unittest.TextTestRunner()
    runner.run(suite)