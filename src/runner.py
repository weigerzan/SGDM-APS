import torch
import logging
import torchvision
import numpy as np
import argparse
import os
import json
import time

class Runner:
    def __init__(self, args, model, train_loader, val_loader, loss_function, optimizer, save_dir='result/cifar10_resnet34', save_file='pmsgd.json', scheduler = None, max_epoch = 100):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epoch = max_epoch
        self.epoch = 0
        self.save_dir = save_dir
        self.save_file = save_file
        self.loss_list = []
        self.epoch_list = []
        self.lr_list = []
        self.eval_acc = []

    def run(self):
        while self.epoch < self.max_epoch:
            aver_loss, aver_lr, epoch = self.train()
            if self.scheduler is not None:
                self.scheduler.step()
            torch.cuda.empty_cache()
            eval_acc = self.eval()
            self.epoch += 1
            self.loss_list.append(aver_loss)
            self.epoch_list.append(epoch)
            self.lr_list.append(aver_lr)
            self.eval_acc.append(eval_acc)
        self.save_res()
    def train(self):
        self.model.train()
        total_lr = 0
        total_loss = 0
        cum_time = 0.0
        cnt_iter = 0
        start_time = time.time()
        for idx, (img, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
            # print(self.save_dir)
            if 'mushrooms' in self.save_dir or 'ijcnn' in self.save_dir or 'rcv1' in self.save_dir or 'w8a' in self.save_dir:
                label = torch.nn.functional.one_hot(label, num_classes=2).float()
            # print(label)
            closure = lambda : self.loss_function(self.model(img), label)
            out = self.model(img)
            loss = self.loss_function(out, label)
            loss.backward()
            self.optimizer.step(closure=closure)
            cnt_iter += 1
            if type(self.optimizer.state['step_size']) is dict:
                if 'lr' in self.optimizer.state_dict()['param_groups'][0].keys():
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                elif 'step_size' in self.optimizer.state_dict()['param_groups'][0].keys():
                    lr = self.optimizer.state_dict()['param_groups'][0]['step_size']
            else:
                lr = self.optimizer.state['step_size']
            if idx % 100 == 0:
                print('optimizer:{}, epoch:{}/{}, step:{}/{}, loss:{}, lr:{}'.format(self.save_file[:-5], \
                    self.epoch, self.max_epoch, idx, len(self.train_loader), loss.item(), lr))
            total_lr += lr
            total_loss += loss.item()
        cum_time = time.time() - start_time
        print('wall-clock time per iteration: {}'.format(cum_time / cnt_iter))
        aver_loss = total_loss / len(self.train_loader)
        aver_lr = total_lr / len(self.train_loader)
        return aver_loss, aver_lr, self.epoch


    def eval(self):
        if isinstance(self.loss_function, torch.nn.MSELoss):
            return 0
        self.model.eval()
        allcnt = 0
        correct = 0
        for idx, (img, label) in enumerate(self.val_loader):
            img = img.cuda()
            label = label.cuda()
            out = self.model(img)
            preds = torch.argmax(out, dim=1)
            allcnt += len(label)
            correct += torch.sum(label == preds)
        print('Presision:{}'.format(correct.item() / allcnt))
        return correct.item() / allcnt

    def save_res(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_file = os.sep.join([self.save_dir, self.save_file])
        res = {'epoch':self.epoch_list, 'loss':self.loss_list, 'lr':self.lr_list, 'eval_acc':self.eval_acc}
        with open(save_file, 'w') as f:
            json.dump(res, f)
            f.close()

            