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
from net.utils.train_utils import adjust_learning_rate, get_learning_rate


class TrainClass:
    def __init__(self):
        self.cfg = Configuration()
        self.log = Logger()
        self.log.open(os.path.join('log.train.txt'), mode='a')
        # net -------------------------------------------------
        self.log.write('** net setting **\n')
        self.mode = 'train_all'
        self.net = MaskRcnnNet(self.cfg).to(self.cfg.device)

        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                   lr=self.cfg.lr / self.cfg.iter_accum,
                                   momentum=0.9,
                                   weight_decay=0.0001
                                   )

        self.start_iter = 1
        self.start_epoch = 0.
        self.rate = 0

        if self.cfg.checkpoint:
            self.load_checkpoint(self.cfg.checkpoint)

        self.init_dataloader()

    def load_checkpoint(self, checkpoint):
        self.net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
        checkpoint_optim = torch.load(checkpoint.replace('_model.pth', '_optimizer.pth'))
        self.start_iter = checkpoint_optim['iter']
        self.start_epoch = checkpoint_optim['epoch']

        self.rate = get_learning_rate(self.optimizer)  # load all except learning rate
        self.optimizer.load_state_dict(checkpoint_optim['optimizer'])
        adjust_learning_rate(self.optimizer, self.rate)

    def save_checkpoint(self, i, epoch):
        model_path = os.path.join('%08d_model.pth' % i)
        optimizer_path = os.path.join('%08d_optimizer.pth' % i)

        torch.save(self.net.state_dict(), model_path)
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'iter': i,
            'epoch': epoch,
        }, optimizer_path)

    def init_dataloader(self):
        self.log.write('** dataset setting **\n')
        self.train_dataset = CocoDataset(self.cfg, self.cfg.data_dir, dataType='train2017', mode='train', transform=train_augment)
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.cfg.batch_size,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_collate)

        self.valid_dataset = CocoDataset(self.cfg, self.cfg.data_dir, dataType='val2017', mode='train', transform=valid_augment)
        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=FixLengthRandomSampler(self.valid_dataset, length=self.cfg.batch_size),
            batch_size=self.cfg.batch_size,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_collate)

        self.log.write('\tlen(train_dataset)  = %d\n' % len(self.train_dataset))
        self.log.write('\tlen(valid_dataset)  = %d\n' % len(self.valid_dataset))
        self.log.write('\tlen(train_loader)   = %d\n' % len(self.train_loader))
        self.log.write('\tlen(valid_loader)   = %d\n' % len(self.valid_loader))
        self.log.write('\tbatch_size  = %d\n' % self.cfg.batch_size)
        self.log.write('\n')

    def get_loss(self):

        rpn_cls_loss = self.net.rpn_cls_loss.detach().cpu().numpy() if self.net.rpn_cls_loss else 0.0
        rpn_reg_loss = self.net.rpn_reg_loss.detach().cpu().numpy() if self.net.rpn_reg_loss else 0.0
        rcnn_cls_loss = self.net.rcnn_cls_loss.detach().cpu().numpy() if self.net.rcnn_cls_loss else 0.0
        rcnn_reg_loss = self.net.rcnn_reg_loss.detach().cpu().numpy() if self.net.rcnn_reg_loss else 0.0
        mask_cls_loss = self.net.mask_cls_loss.detach().cpu().numpy() if self.net.mask_cls_loss else 0.0
        loss = rpn_cls_loss + rpn_reg_loss + rcnn_cls_loss + rcnn_reg_loss + mask_cls_loss

        return np.array((
            loss,
            rpn_cls_loss,
            rpn_reg_loss,
            rcnn_cls_loss,
            rcnn_reg_loss,
            mask_cls_loss,
        ))

    def validate(self):
        self.net.set_mode(self.mode.replace('train', 'valid'))

        test_num = 0
        valid_loss = np.zeros(6, np.float32)
        for inputs, truth_boxes, truth_labels, truth_instances, indices in self.valid_loader:
            with torch.no_grad():
                self.net.loss(inputs.to(self.cfg.device), truth_boxes, truth_labels, truth_instances)

            batch_size = len(indices)
            valid_loss += batch_size * self.get_loss()
            test_num += batch_size

        assert (test_num == len(self.valid_loader.sampler))

        self.net.set_mode(self.mode)
        valid_loss = valid_loss / test_num
        return valid_loss

    def train(self):
        train_loss = np.zeros(6, np.float32)
        valid_loss = np.zeros(6, np.float32)
        batch_loss = np.zeros(6, np.float32)

        start = timer()
        j = 0  # accum counter
        i = 0  # iter  counter

        self.log.write(
            ' rate    iter   epoch  num   | valid    rpnc rpnr    rcnnc rcnnr    mask | train   rpnc rpnr     rcnnc rcnnr    mask | batch   rpnc rpnr     rcnnc rcnnr    mask |  time\n')
        self.log.write(
            '------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

        while i < self.cfg.num_iters:  # loop over the dataset multiple times
            sum_train_loss = np.zeros(6, np.float32)
            sum_train_acc = 0.0
            batch_sum = 0

            self.net.set_mode(self.mode)
            self.optimizer.zero_grad()

            for inputs, truth_boxes, truth_labels, truth_instances, indices in self.train_loader:
                if all(len(b) == 0 for b in truth_boxes):
                    continue
                batch_size = len(indices)
                i = j / self.cfg.iter_accum + self.start_iter
                epoch = (i - self.start_iter) * batch_size * self.cfg.iter_accum / len(self.train_dataset) + self.start_epoch
                num_products = epoch * len(self.train_dataset)
                # validate iter -------------------------------------------------
                if i % self.cfg.iter_valid == 0:
                    valid_loss = self.validate()
                    print('\r', end='', flush=True)
                    self.log.write(
                        '%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %s\n' % (
                            self.rate, i / 1000, epoch, num_products / 1000000,
                            valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],
                            train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                            batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],
                            time_to_str((timer() - start) / 60)))
                    time.sleep(0.01)

                # save checkpoint -------------------------------------------------
                if i % self.cfg.iter_save == 0:
                    self.save_checkpoint(i, epoch)

                # one iteration update  -------------------------------------------------
                loss = self.net.loss(inputs.to(self.cfg.device), truth_boxes, truth_labels, truth_instances)
                loss.backward()

                if j % self.cfg.iter_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)  # gradient clip
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # print statistics  -------------------------------------------------
                batch_acc = 0
                batch_loss = self.get_loss()
                sum_train_loss += batch_loss
                sum_train_acc += batch_acc
                batch_sum += 1
                if i % self.cfg.iter_smooth == 0:
                    train_loss = sum_train_loss / batch_sum
                    sum_train_loss = np.zeros(6, np.float32)
                    sum_train_acc = 0.
                    batch_sum = 0

                print(
                    '\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %0.3f   %0.3f %0.3f   %0.3f %0.3f   %0.3f | %s  %d,%d,%s' % (
                        self.rate, i / 1000, epoch, num_products / 1000000,
                        valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],
                        train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],
                        batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],
                        time_to_str((timer() - start) / 60), i, j, ''), end='', flush=True)
                j = j + 1
            pass  # end of one data loader
        pass  # end of all iterations

        self.log.write('\n')


if __name__ == '__main__':
    # train rpn head
    t = TrainClass()
    print('train RPN head')
    t.mode = 'train_rpn'
    t.train()

    print('train RCNN head')
    t.mode = 'train_rcnn'
    t.train()

    print('train all heads')
    t.mode = 'train_all'
    t.train()

    print('\nsucess!')