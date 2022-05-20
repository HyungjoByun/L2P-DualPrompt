# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import numpy as np
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from analyze.layer_probing import prob_proto_nlp, prob_final_nlp
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
from transformers import ViTFeatureExtractor


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:] = -float('inf') # : 이후 아래처럼 되어있던것 수정
               #dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate_cifar100(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model.net.eval()
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs = feature_extractor([img for img in inputs], return_tensors='pt').pixel_values
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model.forward_model(inputs, k)
            else:
                outputs = model.forward_model(inputs)
            
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
        
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    
    model.net.train(status)
    return accs, accs_mask_classes

# todo: add online learning features to CV tasks.
def train_cifar100(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    results, results_mask_classes = [], []
    
    model_stash = create_stash(model, args, dataset)
    
    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()
    
    print(file=sys.stderr)
    start_time = time.time()
    for t in range(dataset.N_TASKS):
        model.net.train()
        if t == 0: train_loader, test_loader = dataset.get_data_loaders(download=True)
        else: train_loader, test_loader = dataset.get_data_loaders(download=False)

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        
        model.init_opt(args)
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    #feature extractor는 vit pretraining때와 같도록 image size등을 변경
                    inputs = feature_extractor([img for img in inputs], return_tensors='pt').pixel_values
                    #inputs = (inputs+1)/2
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    #loss = model.observe(inputs, labels, not_aug_inputs, logits)
                    loss = model.observe(inputs, labels)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs = feature_extractor([img for img in inputs], return_tensors='pt').pixel_values
                    #inputs = (inputs+1)/2
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    #loss = model.observe(inputs, labels, not_aug_inputs)
                    loss = model.observe(inputs, labels,dataset,t)
                # if (i % 40) == 0: print(labels) #! debug: labels per task
                progress_bar(i, len(train_loader), epoch, t, loss)
                
                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
                
                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0

        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0
        
        if hasattr(model, 'end_task'):
            model.end_task(dataset)
        
        accs = evaluate_cifar100(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        print("")
        print(accs[0])
        print(accs[1])
        
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        
        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    running_time = time.time() - start_time
    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_running_time(running_time)
        csv_logger.add_forgetting(results, results_mask_classes)
    
    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))