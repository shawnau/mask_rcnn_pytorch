import os
import numpy as np
import time
from timeit import default_timer as timer
from torch import optim
from torch.utils.data import DataLoader

from configuration import Configuration
from loader.coco.dataset import CocoDataset, train_augment, valid_augment, train_collate
from net.mask_rcnn import MaskRcnnNet

from loader.sampler import *
from net.utils.file import Logger, time_to_str


def validate(cfg, net, valid_loader):
    test_num = 0
    valid_loss = np.zeros(6, np.float32)

    for inputs, truth_boxes, truth_labels, truth_instances, indices in valid_loader:
        with torch.no_grad():
            net(inputs.to(cfg.device), truth_boxes, truth_labels, truth_instances)
            loss = net.loss()

        batch_size = len(indices)
        valid_loss += batch_size * np.array((
            loss.cpu().data.numpy(),
            net.rpn_cls_loss.cpu().data.numpy(),
            net.rpn_reg_loss.cpu().data.numpy(),
            net.rcnn_cls_loss.cpu().data.numpy() if net.rcnn_cls_loss else 0.0,
            net.rcnn_reg_loss.cpu().data.numpy() if net.rcnn_reg_loss else 0.0,
            net.mask_cls_loss.cpu().data.numpy() if net.mask_cls_loss else 0.0,
        ))
        test_num += batch_size

    assert (test_num == len(valid_loader.sampler))
    valid_loss = valid_loss / test_num
    return valid_loss


def run_train():
    cfg = Configuration()

    log = Logger()
    log.open(os.path.join('log.train.txt'), mode='a')

    # net -------------------------------------------------
    log.write('** net setting **\n')
    net = MaskRcnnNet(cfg).to(cfg.device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=cfg.lr / cfg.iter_accum,
                          momentum=0.9,
                          weight_decay=0.0001
                          )

    start_iter = 0
    start_epoch = 0.

    # dataset -------------------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = CocoDataset(cfg, cfg.data_dir, dataType='train2017', mode='train', transform=train_augment)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate)

    valid_dataset = CocoDataset(cfg, cfg.data_dir, dataType='val2017', mode='train', transform=valid_augment)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=FixLengthRandomSampler(valid_dataset, length=cfg.batch_size),
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate)

    log.write('\tlen(train_dataset)  = %d\n' % len(train_dataset))
    log.write('\tlen(valid_dataset)  = %d\n' % len(valid_dataset))
    log.write('\tlen(train_loader)   = %d\n' % len(train_loader))
    log.write('\tlen(valid_loader)   = %d\n' % len(valid_loader))
    log.write('\tbatch_size  = %d\n' % cfg.batch_size)
    log.write('\titer_accum  = %d\n' % cfg.iter_accum)
    log.write('\tbatch_size*iter_accum  = %d\n' % (cfg.batch_size * cfg.iter_accum))
    log.write('\n')

    # start training here! -------------------------------------------------
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer))
    log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])

    log.write(' images_per_epoch = %d\n\n' % len(train_dataset))
    log.write(
        ' rate    iter   epoch  num   | valid    rpnc rpnr    rcnnc rcnnr    mask | train   rpnc rpnr     rcnnc rcnnr    mask | batch   rpnc rpnr     rcnnc rcnnr    mask |  time\n')
    log.write(
        '------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)
    rate = 0

    start = timer()
    j = 0  # accum counter
    i = 1  # iter  counter

    while i < cfg.num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6, np.float32)
        sum_train_acc = 0.0
        batch_sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, indices in train_loader:
            if all(len(b) == 0 for b in truth_boxes):
                continue
            batch_size = len(indices)
            i = j / cfg.iter_accum + start_iter
            epoch = (i - start_iter) * batch_size * cfg.iter_accum / len(train_dataset) + start_epoch
            num_products = epoch * len(train_dataset)
            # validate iter -------------------------------------------------
            if i % cfg.iter_valid == 0:
                net.set_mode('valid')
                valid_loss = validate(cfg, net, valid_loader)
                net.set_mode('train')

                print('\r', end='', flush=True)
                log.write(
                    '%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %s\n' % (
                        rate, i / 1000, epoch, num_products / 1000000,
                        valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],
                        # valid_acc,
                        train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                        # train_acc,
                        batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],
                        # batch_acc,
                        time_to_str((timer() - start) / 60)))
                time.sleep(0.01)
            # save checkpoint_dir -------------------------------------------------
            if i % cfg.iter_save == 0:
                model_path = os.path.join('%08d_model.pth' % i)
                optimizer_path = os.path.join('%08d_optimizer.pth' % i)

                torch.save(net.state_dict(), model_path)
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                }, optimizer_path)

            # one iteration update  -------------------------------------------------
            net(inputs.to(cfg.device), truth_boxes, truth_labels, truth_instances)
            loss = net.loss()

            # accumulated update
            loss.backward()
            if j % cfg.iter_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)  # gradient clip
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  -------------------------------------------------
            batch_acc = 0
            batch_loss = np.array((
                loss.cpu().data.numpy(),
                net.rpn_cls_loss.cpu().data.numpy(),
                net.rpn_reg_loss.cpu().data.numpy(),
                net.rcnn_cls_loss.cpu().data.numpy(),
                net.rcnn_reg_loss.cpu().data.numpy(),
                net.mask_cls_loss.cpu().data.numpy(),
            ))
            sum_train_loss += batch_loss
            sum_train_acc += batch_acc
            batch_sum += 1
            if i % cfg.iter_smooth == 0:
                train_loss = sum_train_loss / batch_sum
                sum_train_loss = np.zeros(6, np.float32)
                sum_train_acc = 0.
                batch_sum = 0

            print(
                '\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %s  %d,%d,%s' % (
                    rate, i / 1000, epoch, num_products / 1000000,
                    valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],
                    # valid_acc,
                    train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                    # train_acc,
                    batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],
                    # batch_acc,
                    time_to_str((timer() - start) / 60), i, j, ''), end='', flush=True)
            j = j + 1
        pass  # end of one data loader
    pass  # end of all iterations

    log.write('\n')


if __name__ == '__main__':
    run_train()

    print('\nsucess!')