import torch.nn as nn
import torch.nn.functional as F


class SEScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class Bottleneck(nn.Module):
    """
    SE-ResNeXt 50 BottleNeck
    """
    def __init__(self, i_ch, o_ch, stride=1, groups=32):
        """
        :param i_ch: input tensor channels
        :param o_ch: output tensor channels in each stage
        :param stride: the 3x3 kernel in a stage may set to 2
                       only in each group of stages' 1st stage
        """
        super(Bottleneck, self).__init__()
        s_ch = o_ch // 2  # stage channel
        self.conv1 = nn.Conv2d(i_ch, s_ch, kernel_size=1, padding=0, stride=1,      bias=False)
        self.bn1 = nn.BatchNorm2d(s_ch)
        self.conv2 = nn.Conv2d(s_ch, s_ch, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(s_ch)
        self.conv3 = nn.Conv2d(s_ch, o_ch, kernel_size=1, padding=0, stride=1,      bias=False)
        self.bn3 = nn.BatchNorm2d(o_ch)
        self.scale = SEScale(o_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()  # empty Sequential module returns the original input

        if stride != 1 or i_ch != o_ch:  # tackle input/output size/channel mismatch during shortcut add
            self.shortcut = nn.Sequential(
                nn.Conv2d(i_ch, o_ch, kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(o_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.scale(out)*out + self.shortcut(x)
        out = self.relu(out)

        return out


class LateralBlock(nn.Module):
    """
    Feature Pyramid LateralBlock
    """
    def __init__(self, c_planes):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes, 256, 1)
        self.smooth = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, c, p):
        """
        :param c: c layer before conv 1x1
        :param p: p layer before upsample
        :return: conv3x3( conv1x1(c) + upsample(p) )
        """
        _, _, H, W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2, mode='nearest')
        # p = F.upsample(p, size=(H, W), mode='bilinear')
        p = p[:, :, :H, :W] + c
        p = self.smooth(p)
        return p


class SEResNeXtFPN(nn.Module):
    def _make_conv1(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # (224-7+2*3) // 2 +1 = 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # shrink to 1/4 of original size
        )

    def _make_stage(self, i_ch, o_ch, num_blocks, stride=1):
        """
        making conv_2, conv_3, conv_4, conv_5
        :param i_ch: channels of input  tensor
        :param o_ch: channels of output tensor
        :param num_blocks: repeats of bottleneck
        :param stride: stride of the 3x3 conv layer of each bottleneck
        :return:
        """
        layers = []
        layers.append(Bottleneck(i_ch, o_ch, stride))  # only the first stage in the module need stride=2
        for i in range(1, num_blocks):
            layers.append(Bottleneck(o_ch, o_ch))
        return nn.Sequential(*layers)

    def __init__(self, num_list):
        super(SEResNeXtFPN, self).__init__()

        # bottom up layers/stages
        self.conv1 = self._make_conv1()   # 3 -> (7x7 x 64) -> 64 -> BN -> ReLU -> MaxPool ->64
        self.conv2_x = self._make_stage(64,   256,  num_list[0], stride=1)
        self.conv3_x = self._make_stage(256,  512,  num_list[1], stride=2)
        self.conv4_x = self._make_stage(512,  1024, num_list[2], stride=2)
        self.conv5_x = self._make_stage(1024, 2048, num_list[3], stride=2)
        # top down layers
        self.layer_p5 = nn.Conv2d(2048, 256, 1)
        self.layer_p4 = LateralBlock(1024)  # takes p5 and c4 (1024)
        self.layer_p3 = LateralBlock(512)
        self.layer_p2 = LateralBlock(256)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2_x(c1)
        c3 = self.conv3_x(c2)
        c4 = self.conv4_x(c3)
        c5 = self.conv5_x(c4)

        p5 = self.layer_p5(c5)
        p4 = self.layer_p4(c4, p5)
        p3 = self.layer_p3(c3, p4)
        p2 = self.layer_p2(c2, p3)

        features = [p2, p3, p4, p5]
        return features
