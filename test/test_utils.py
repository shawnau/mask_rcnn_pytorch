import unittest

import numpy as np

from net.lib.box_overlap.cython_box_overlap import cython_box_overlap


class TestCythonOverlap(unittest.TestCase):
    def setUp(self):
        self.boxes = np.array([[-0.5, -0.5, 0.5, 0.5],
                               [-1.0, -2.0, 2.0, 1.0]]
                              ).astype(np.float32)

        self.query = np.array([[-1.0, -1.0, 1.0, 1.0],
                               [-1.5, -1.5, 1.5, 1.5],
                               [-2.0, -2.0, 2.0, 2.0]]
                              ).astype(np.float32)

    def test_forward(self):
        overlap = cython_box_overlap(self.boxes, self.query)
        print(overlap)


if __name__ == '__main__':
    unittest.main()