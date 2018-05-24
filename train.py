import torch
from torch import optim
from torch.utils.data import DataLoader

from configuration import Configuration
from net.mask_rcnn import MaskRcnnNet

from loader.dsb2018.train_utils import *


def run_train():
    cfg = Configuration()
    net = MaskRcnnNet(cfg).to(cfg.device)
    # loader
    train_dataset = ScienceDataset(cfg, mode='train', transform=train_augment)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=make_collate)
    # optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=cfg.lr/cfg.iter_accum,
                          momentum=0.9,
                          weight_decay=0.0001
                          )

    i = 0
    j = 0
    while i < cfg.num_iters:
        net.set_mode('train')
        optimizer.zero_grad()

        for images, truth_boxes, truth_labels, truth_instances, indices in train_loader:

            if all(len(b) == 0 for b in truth_boxes):
                continue
            i += j / cfg.iter_accum

            net(images.to(cfg.device), truth_boxes, truth_labels, truth_instances)
            loss = net.loss()
            loss.backward()

            if j % cfg.iter_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1) # gradient clip
                optimizer.step()
                optimizer.zero_grad()
            j += 1

            print("iter: ", j, "%.4f, %.4f, %.4f, %.4f, %.4f"%(net.rpn_cls_loss.detach().cpu().numpy(),
                                                               net.rpn_reg_loss.detach().cpu().numpy(),
                                                               net.rcnn_cls_loss.detach().cpu().numpy(),
                                                               net.rcnn_reg_loss.detach().cpu().numpy(),
                                                               net.mask_cls_loss.detach().cpu().numpy()) )
            if j%100 == 0:
                torch.save(net.state_dict(), "%s.pth"%j)


if __name__ == '__main__':
    run_train()