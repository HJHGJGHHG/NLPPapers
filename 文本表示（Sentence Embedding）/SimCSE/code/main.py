import os
import time
import torch
import random
import numpy as np
from os.path import join
from tqdm import tqdm
from loguru import logger
from transformers import BertModel, BertConfig, BertTokenizer

from config import args_init
from train_eval_test import train, evaluate
from load_data import load_iter, load_predict_iter
from models import SimCSEModel, simcse_unsup_loss, simcse_sup_loss


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(args):
    # preparations
    set_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    args.output_path = join(args.output_path, args.train_mode,
                            'bsz-{}-lr-{}-dropout-{}-mlm-{}'.format(args.batch_size_train, args.lr, args.dropout,
                                                                    args.do_mlm))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    # load model
    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg", 'cls-mlp'], \
        'pooler should in ["cls", "pooler", "last-avg", "first-last-avg", "cls-mlp"]'
    model = SimCSEModel(args).to(args.device)
    
    # load data
    if args.do_train:
        # 加载数据集
        train_iter, dev_iter = load_iter(args, tokenizer)
        train(args, model, train_iter, dev_iter)
    if args.do_predict:
        test_iter = load_predict_iter(args, tokenizer)
        corrcoef = evaluate(args, model, test_iter)
        logger.info('Average Spearman’s correlation on test dataset: {}'.format(corrcoef))


if __name__ == '__main__':
    parser = args_init()
    args = parser.parse_args([])
    main(args)
