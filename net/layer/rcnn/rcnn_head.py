import torch.nn as nn
import torch.nn.functional as F

class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RcnnHead, self).__init__()
        self.num_classes = cfg.num_classes
        self.crop_size   = cfg.rcnn_crop_size

        self.fc1 = nn.Linear(in_channels*self.crop_size*self.crop_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.logit = nn.Linear(1024, self.num_classes)
        self.delta = nn.Linear(1024, self.num_classes*4)

    def forward(self, crops):
        """
        :param crops: cropped feature maps
            shape = (B, C, crop_size, crop_size)
        :return:
            logits: (B, num_classes)
            delta: (B, num_classes*4)
        """
        # flatten each cropped feature map into C*crop_size*crop_size
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas