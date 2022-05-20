# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import torch
import numpy as np
import random
from datasets import NAMES as DATASET_NAMES
from datasets import get_dataset

from models import get_all_models
from models import get_model

from argparse import ArgumentParser
from utils.args import add_management_args

from utils.training import train_nlp
from utils.training_glue import train_glue
from utils.training_cifar100 import train_cifar100
from utils.best_args import best_args
from utils.conf import set_random_seed

def main():
    parser = ArgumentParser(description='pseudoCL', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    
    args = parser.parse_known_args()[0]
    
    mod = importlib.import_module('models.' + args.model)
    
    
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()
    
    if args.seed is not None:
        set_random_seed(args.seed)
    
    if args.model == 'mer': setattr(args, 'batch_size', 1)
    
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform()) #모델 객체 생성(init)

    #아직 dataset구현 안했는데 일단 임시로 적음. seq-pmnist로 바꾸면 작동 가능(exp_2_1_pt.py참고)
    if dataset.NAME == "seq-cifar100":
        train_cifar100(model, dataset, args)
        return
    
    if dataset.NAME == "seq-glue":
        train_glue(model, dataset, args)
        return

    if args.area == "NLP":
        train_nlp(model, dataset, args)
        return


if __name__ == '__main__':
    main()
