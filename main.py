from src.dataset import load_cifar10, load_cifar100, load_svm_dataset, load_matrix_fac
from src.runner import Runner
from optim.SGDM_APS import SGDM_APS
import torch
import logging
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from models.base_classifiers import get_classifier
import os
import json


def get_dataset_model_loss(task):
    if task == 'ijcnn':
        trainset = load_svm_dataset(dataset_name='ijcnn', train=True)
        valset = load_svm_dataset(dataset_name='ijcnn', train=False)
        model = get_classifier('linear', trainset)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean") # logistic loss
    elif task == 'mushrooms':
        trainset = load_svm_dataset(dataset_name='mushrooms', train=True)
        valset = load_svm_dataset(dataset_name='mushrooms', train=False)
        model = get_classifier('linear', trainset)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean") # logistic loss
    elif task == 'rcv1':
        trainset = load_svm_dataset(dataset_name='rcv1', train=True)
        valset = load_svm_dataset(dataset_name='rcv1', train=False)
        model = get_classifier('linear', trainset)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean") # logistic loss
    elif task == 'w8a':
        trainset = load_svm_dataset(dataset_name='w8a', train=True)
        valset = load_svm_dataset(dataset_name='w8a', train=False)
        model = get_classifier('linear', trainset)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean") # logistic loss
    elif task == 'matrix_1':
        trainset = load_matrix_fac(train=True, datadir='dataset')
        valset = load_matrix_fac(train=False, datadir='dataset')
        model = get_classifier('matrix_fac_1', trainset)
        loss_function = torch.nn.MSELoss()
    elif task == 'matrix_4':
        trainset = load_matrix_fac(train=True, datadir='dataset')
        valset = load_matrix_fac(train=False, datadir='dataset')
        model = get_classifier('matrix_fac_4', trainset)
        loss_function = torch.nn.MSELoss()
    elif task == 'matrix_10':
        trainset = load_matrix_fac(train=True, datadir='dataset')
        valset = load_matrix_fac(train=False, datadir='dataset')
        model = get_classifier('matrix_fac_10', trainset)
        loss_function = torch.nn.MSELoss()
    elif task == 'cifar10':
        trainset = load_cifar10(train=True)
        valset = load_cifar10(train=False)
        model = get_classifier('resnet34', trainset)
        loss_function = torch.nn.CrossEntropyLoss()
    elif task == 'cifar100':
        trainset = load_cifar100(train=True)
        valset = load_cifar100(train=False)
        model = get_classifier('resnet34_100', trainset)
        loss_function = torch.nn.CrossEntropyLoss()
    elif task == 'cifar10_densenet':
        trainset = load_cifar10(train=True)
        valset = load_cifar10(train=False)
        model = get_classifier('densenet121', trainset)
        loss_function = torch.nn.CrossEntropyLoss()
    elif task == 'cifar100_densenet':
        trainset = load_cifar100(train=True)
        valset = load_cifar100(train=False)
        model = get_classifier('densenet121_100', trainset)
        loss_function = torch.nn.CrossEntropyLoss()
    return trainset, valset, model, loss_function

def test_SGDM_APS(arg):
    task = args.task
    trainset, valset, model, loss_function = get_dataset_model_loss(task)
    train_loader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.bs, shuffle=False)
    optimizer = SGDM_APS(model.parameters(), momentum=arg.beta, warmup_steps=len(trainset), c=args.c)
    model = model.cuda()
    Trainer = Runner(args, model, train_loader, val_loader, loss_function, optimizer,\
        save_dir="result/{}".format(task),  save_file='SGDM-APS.json'.format(task), max_epoch = args.max_epoch)
    Trainer.run()

def main(args):
    test_SGDM_APS(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--c', type=float, default=0.2)
    parser.add_argument('--omega', type=float, default=2)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--task', type=str, default='cifar10', \
        choices=['ijcnn', 'mushrooms', 'rcv1', 'w8a', 'matrix_1', 'matrix_4', 'matrix_10', 'cifar10', 'cifar100', 'cifar10_densenet', 'cifar100_densenet'])
    args = parser.parse_args() 
    main(args)


