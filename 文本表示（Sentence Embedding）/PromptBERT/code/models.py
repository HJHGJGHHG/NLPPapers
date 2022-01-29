import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertPredictionHeadTransform


class MLPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        
        return x


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    if cls.args.train_mode == 'supervised':
        cls.mlp = BertPredictionHeadTransform(config)
    else:
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.args.temp)
    cls.init_weights()


def cl_forward(cls,
               encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               return_dict=None,
               ):
    def get_delta(template_token, length=50):
        with torch.set_grad_enabled(True):
            device = input_ids.device
            d_input_ids = torch.Tensor(template_token).repeat(length, 1).to(device).long()
            d_inputs_embeds = None
            d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(length, 1).long()
            d_position_ids[:, len(cls.bs) + 1:] += torch.arange(length).to(device).unsqueeze(-1)
            m_mask = d_input_ids == cls.mask_token_id
            outputs = encoder(input_ids=d_input_ids if d_inputs_embeds is None else None,
                              inputs_embeds=d_inputs_embeds,
                              position_ids=d_position_ids, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            delta = last_hidden[m_mask]
            template_len = d_input_ids.shape[1]
            if cls.args.train_mode == 'supervised':
                delta = cls.mlp(delta)
            return delta, template_len
    
    if cls.args.denoising:
        delta1, template_len1 = get_delta([cls.template1])  # [50, 768], 10
        if len(cls.args.template2) > 0:
            delta2, template_len2 = get_delta([cls.template2])  # [50, 768], 10
    
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    sentence_num = input_ids.size(1)
    
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)
    
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_hidden_states=False,
        return_dict=True,
    )
    
    # Pooling
    last_hidden = outputs.last_hidden_state
    pooler_output = last_hidden[input_ids == cls.mask_token_id]
    
    if cls.args.denoising:
        if cls.args.train_mode == 'supervised':
            pooler_output = cls.mlp(pooler_output)
        
        if len(cls.args.template2) > 0:
            pooler_output = pooler_output.view(batch_size, sentence_num, -1)
            attention_mask = attention_mask.view(batch_size, sentence_num, -1)
            blen = attention_mask.sum(-1) - template_len1
            pooler_output[:, 0, :] -= delta1[blen[:, 0]]
            blen = attention_mask.sum(-1) - template_len2
            pooler_output[:, 1, :] -= delta2[blen[:, 1]]
            if sentence_num == 3:
                pooler_output[:, 2, :] -= delta2[blen[:, 2]]
        else:
            blen = attention_mask.sum(-1) - template_len1
            pooler_output -= delta1[blen]
    
    pooler_output = pooler_output.view(batch_size * sentence_num, -1)
    
    pooler_output = pooler_output.view((batch_size, sentence_num, pooler_output.size(-1)))  # (bs, num_sent, hidden)
    
    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    
    # Hard negative
    if sentence_num == 3:
        z3 = pooler_output[:, 2]
    
    logits = (z1, z2) if sentence_num == 2 else (z1, z2, z3)
    
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if sentence_num >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
    
    loss_fct = nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)
    
    # Calculate loss with hard negatives
    if sentence_num == 3:
        # Note that weights are actually logits of weights
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [0.0] * (
                    z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(input_ids.device)
        cos_sim = cos_sim + weights
    
    loss = loss_fct(cos_sim, labels)
    
    output = {
        'loss': loss,
        'logits': logits,
    }
    return output


class BertForCL(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.bert = BertModel(config)
        cl_init(self, config)
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                return_dict=None,
                score=None,
                ):
        return cl_forward(self, self.bert,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          return_dict=return_dict,
                          )
