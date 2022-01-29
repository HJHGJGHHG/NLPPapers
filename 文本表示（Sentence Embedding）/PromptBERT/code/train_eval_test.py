import torch
import numpy as np
import torch.nn.functional as F

from os.path import join
from tqdm import tqdm
from loguru import logger
from scipy.stats import spearmanr
from transformers import AdamW, get_scheduler


def training_preparations(args, model, train_iter):
    if args.weight_decay is not None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay,
             },
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.use_lr_scheduler:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.epochs * len(train_iter) * args.warmup_ratio,
            num_training_steps=args.epochs * len(train_iter), )
        return optimizer, lr_scheduler
    
    return optimizer


def train(args, model, train_iter, dev_iter):
    if args.use_lr_scheduler:
        optimizer, lr_scheduler = training_preparations(args, model, train_iter)
    else:
        optimizer = training_preparations(args, model, train_iter)
    logger.info("start training")
    best = 0
    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(tqdm(train_iter)):
            model.train()
            global_step = (epoch - 1) * len(train_iter) + batch_idx
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch)['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.use_lr_scheduler:
                lr_scheduler.step()
            global_step += 1
            
            # evaluate
            if global_step % args.eval_step == 0:
                corrcoef = evaluate(args, model, dev_iter)
                logger.info(
                    'loss: {}, corrcoef: {} in global_step {} epoch {}'.format(loss, corrcoef, global_step, epoch))
                model.train()
                if best < corrcoef:
                    best = corrcoef
                    torch.save(model.state_dict(), join(args.output_path, 'simcse.pt'))
                    logger.info(
                        'get higher corrcoef: {} in global_step {} epoch {}, save model'.format(best, global_step,
                                                                                                epoch))
    return best

def evaluate(args, model, dev_iter):
    model.eval()
    sim_tensor = torch.tensor([], device=args.device)
    label_array = np.array([])
    with torch.no_grad():
        for batch in dev_iter:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            logits = model(**batch)['logits']
            # compute cosine similarity
            sim = F.cosine_similarity(logits[0], logits[1], dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(torch.squeeze(batch['score']).cpu()))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
