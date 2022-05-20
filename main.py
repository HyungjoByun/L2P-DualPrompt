# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from datasets import get_dataset

from models import get_all_models
from models import get_model

from argparse import ArgumentParser
from utils.args import add_management_args

from training_cifar100 import train_cifar100
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
    
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform()) #모델 객체 생성(init)

    train_cifar100(model, dataset, args)


if __name__ == '__main__':
    main()
