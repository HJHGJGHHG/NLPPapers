import os
import torch
import random
import numpy as np
from datasets import load_metric

from model import SiamLSTM
from load_data import load_iter
from train_val_test import train, eval_model



def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default='./data/', help="data cache dir.")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("-embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_size", default=50, type=int)
    parser.add_argument("--epochs", default=10, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="weight decay (default: 1e-2)", dest="weight_decay", )
    parser.add_argument("--lr_scheduler_type", default="linear", help="the scheduler type to use.",
                        choices=[
                            "linear",
                            "cosine",
                            "cosine_with_restarts",
                            "polynomial",
                            "constant",
                            "constant_with_warmup",
                        ], )
    parser.add_argument("--warmup_ratio", default=0.1, type=int,
                        help="number of steps for the warmup in the lr scheduler.", )
    parser.add_argument("--test_only", help="only test the model", action="store_true", )
    parser.add_argument("--seed", default=1234, type=int, help="a seed for reproducible training.")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args([])
    set_seed(args.seed)
    data, train_iter, val_iter, test_iter = load_iter(args)
    metric = load_metric('metric.py', 'stsb')
    
    model = SiamLSTM(data['max_seq_len'], data['embeddings'], args)
    model.to(args.device)
    
    train(args, model, train_iter, val_iter, metric)
    