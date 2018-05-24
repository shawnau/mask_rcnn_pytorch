import torch
from torch import nn
import torch.nn.functional as F


class RpnMultiHead(nn.Module):
    """
    N-way RPN head
    """
    def __init__(self, cfg, in_channels):
        """
        :param cfg: <todo> better config file
        :param in_channels:
        """
        super(RpnMultiHead, self).__init__()

        self.num_scales = len(cfg.rpn_scales)
        self.num_bases = [len(b) for b in cfg.rpn_base_apsect_ratios]

        self.convs  = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.deltas = nn.ModuleList()

        for l in range(self.num_scales):
            channels = in_channels*2
            self.convs.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            self.logits.append(
                nn.Sequential(
                    nn.Conv2d(channels, 2*self.num_bases[l], kernel_size=3, padding=1),
                )
            )
            self.deltas.append(
                nn.Sequential(
                    nn.Conv2d(channels, 4*2*self.num_bases[l], kernel_size=3, padding=1),
                )
            )

    def forward(self, fs):
        """
        :param fs: [p2, p3, p4, p5]
        :return:
            logits_flat:    base: 1 : 1  1 : 2  2 : 1
                        p2[0][0]: [f_submit, b] [f_submit, b] [f_submit, b],
                        p2[0][1]: [f_submit, b] [f_submit, b] [f_submit, b],
                                         ...
                        p2[16][16]:
                           ...
                        p5[:][:]
            shape: (B, N, 2)

            f_submit = foreground prob
            b = background prob

            deltas_flat:    base:   1 : 1    1 : 2    2 : 1
                        p2[0][0]: [df, db] [df, db] [df, db]
                        p2[0][1]: [df, db] [df, db] [df, db]
                                         ...
                        p2[16][16]:
                           ...
                        p5[:][:]
            shape: (B, N, 2, 4)

            df = foreground deltas
            db = background deltas
            in which df, db = [cx, cy, w, h]

            a total of N = (16*16*3 + 32*32*3 + 64*64*3 + 128*128*3) = 65280 proposals in an input
        """
        batch_size = len(fs[0])

        logits_flat = []
        deltas_flat = []
        for l in range(self.num_scales):  # apply multibox head to feature maps
            f = fs[l]
            f = F.relu(self.convs[l](f))
            f = F.dropout(f, p=0.5, training=self.training)

            logit = self.logits[l](f)
            delta = self.deltas[l](f)

            logit_flat = logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2, 4)

            logits_flat.append(logit_flat)
            deltas_flat.append(delta_flat)

        logits_flat = torch.cat(logits_flat, 1)
        deltas_flat = torch.cat(deltas_flat, 1)

        return logits_flat, deltas_flat
