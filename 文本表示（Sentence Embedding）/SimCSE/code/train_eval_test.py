import torch
import numpy as np
import torch.nn.functional as F

from os.path import join
from tqdm import tqdm
from loguru import logger
from scipy.stats import spearmanr
from transformers import AdamW, get_scheduler

from models import simcse_unsup_loss, simcse_sup_loss


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
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = batch['input_ids'].shape[-1]
            input_ids = batch['input_ids'].view(-1, sql_len).to(args.device)
            attention_mask = batch['attention_mask'].view(-1, sql_len).to(args.device)
            token_type_ids = batch['token_type_ids'].view(-1, sql_len).to(args.device)
            if args.do_mlm:
                mlm_input_ids = batch['mlm_input_ids'].view(-1, sql_len).to(args.device)
                mlm_labels = batch['mlm_labels'].view(-1, sql_len).to(args.device)
                losses = model(input_ids, attention_mask, token_type_ids, mlm_input_ids, mlm_labels)
            else:
                losses = model(input_ids, attention_mask, token_type_ids)
            
            if args.train_mode == 'unsupervise':
                loss = simcse_unsup_loss(losses['cl_logits'], args.device, args.temp)
            else:
                loss = simcse_sup_loss(losses['cl_logits'], args.device, args.temp)
            if args.do_mlm:
                loss_fct = torch.nn.CrossEntropyLoss()
                mlm_loss = loss_fct(losses['prediction_scores'].view(-1, model.config.vocab_size), mlm_labels.view(-1))
                loss += args.mlm_weight * mlm_loss
            
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


def evaluate(args, model, dev_iter):
    model.eval()
    sim_tensor = torch.tensor([], device=args.device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dev_iter:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(args.device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(args.device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(args.device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)['cl_logits']
            
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(args.device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(args.device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(args.device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)['cl_logits']
            
            # compute cosine similarity
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
