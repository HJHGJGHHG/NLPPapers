import torch
import logging
import torch.utils.data

from config import get_args_parser
from utils import preparations, set_seed, mkdir
from train_eval import train, evaluate

# ----------logger-----------
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("stsb_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def main(args):
    set_seed(args.seed)
    train_iter, val_iter, model, criterion, optimizer, lr_scheduler, metric = preparations(args)
    if args.output_dir:
        mkdir(args.output_dir)
    logger.info(str(args))
    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
        
    if args.test_only:
        evaluate(args, model, val_iter, metric, logger)
        return

    pearson, spearmanr, combined_score = train(args, model, train_iter, val_iter, metric, optimizer, lr_scheduler, logger, scaler)
    
    print(" * Pearson {:.4f}".format(pearson))
    print(" * Spearmanr {:.10f}".format(spearmanr))
    print(" * Combined Score: {0:.4f}".format(combined_score))
    
if __name__ == '__main__':
    args = get_args_parser().parse_args([])
    main(args)