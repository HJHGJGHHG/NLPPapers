# -*- coding: utf-8 -*-
# Reference: https://github.com/songyouwei/ABSA-PyTorch

import torch
import torch.nn as nn


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        使LSTM支持一个batch中含有长度不同的句子，基于pack_padded_sequence方法实现
        参考：https://zhuanlan.zhihu.com/p/34418001
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx].to('cpu')
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)
        
        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)
            
            return out, (ht, ct)


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(args.hidden_dim, args.polarities_dim)
    
    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out


class TD_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TD_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)
    
    def forward(self, inputs):
        x_l, x_r = inputs[0], inputs[1]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out


class TC_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TC_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)
    
    def forward(self, inputs):
        # Get the target and its length(target_len)
        x_l, x_r, target = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1, dtype=torch.float)[:, None, None]
        x_l, x_r, target = self.embed(x_l), self.embed(x_r), self.embed(target)
        v_target = torch.div(target.sum(dim=1, keepdim=True),
                             target_len)  # v_{target} in paper: average the target words
        
        # the concatenation of word embedding and target vector v_{target}:
        x_l = torch.cat(
            (x_l, torch.cat(([v_target] * x_l.shape[1]), 1)),
            2
        )
        x_r = torch.cat(
            (x_r, torch.cat(([v_target] * x_r.shape[1]), 1)),
            2
        )
        
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out
