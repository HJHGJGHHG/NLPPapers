import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class SimCSEModel(nn.Module):
    """Simcse Model"""
    
    def __init__(self, args):
        super(SimCSEModel, self).__init__()
        self.args = args
        config = BertConfig.from_pretrained(args.pretrain_model_path)
        config.attention_probs_dropout_prob = args.dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = args.dropout
        self.config = config
        self.pooling = args.pooler
        if self.pooling == 'cls-mlp':
            self.mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.bert = BertModel.from_pretrained(args.pretrain_model_path, config=config)
        if args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
    
    def forward(self, input_ids, attention_mask, token_type_ids, mlm_input_ids=None, mlm_labels=None):
        output = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            output_hidden_states=True,
            return_dict=True)
        if self.pooling == 'cls':
            cl_logits = output.last_hidden_state[:, 0]  # [batch, 768]
        elif self.pooling == 'cls-mlp':
            cl_logits = self.mlp(output.last_hidden_state[:, 0])  # [batch, 768]
        elif self.pooling == 'pooler':
            cl_logits = output.pooler_output  # [batch, 768]
        elif self.pooling == 'last-avg':
            last = output.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            cl_logits = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        else:  # self.pooling == 'first-last-avg'
            first = output.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = output.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            cl_logits = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        
        if (mlm_input_ids is not None) and (mlm_labels is not None):
            mlm_output = self.bert(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=True)
            prediction_scores = self.lm_head(mlm_output.last_hidden_state)
            
            return {'cl_logits': cl_logits, 'prediction_scores': prediction_scores}
        
        return {'cl_logits': cl_logits}


def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def simcse_sup_loss(y_pred, device, temp=0.05):
    """
    有监督损失函数
    """
    similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
    row = torch.arange(0, y_pred.shape[0], 3)
    col = torch.arange(0, y_pred.shape[0])
    col = col[col % 3 != 0]
    
    similarities = similarities[row, :]
    similarities = similarities[:, col]
    similarities = similarities / temp
    
    y_true = torch.arange(0, len(col), 2, device=device)
    loss = F.cross_entropy(similarities, y_true)
    return loss
