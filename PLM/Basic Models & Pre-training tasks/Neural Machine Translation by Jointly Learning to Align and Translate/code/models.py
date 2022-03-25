from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        """
        :param emb_dim: Embedding dimension of input (x_t)
        :param enc_hid_dim: hidden layer dimension of Encoder
        :param dec_hid_dim: hidden layer dimension of Decoder
        :param dropout:
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x = [seq_len, batch_size]
        """
        x = x.transpose(0, 1)  # x = [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x)).transpose(0, 1)  # embedded = [seq_len, batch_size, emb_dim]
        
        # enc_output = [seq_len, batch_size, hid_dim * num_directions]
        # enc_hidden = [n_layers * num_directions, batch_size, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)  # if h_0 is not give, it will be set 0 acquiescently
        
        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer
        
        # enc_hidden [-2, :, : ] is the last of the forwards RNN
        # enc_hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2, :, :], enc_hidden[-1, :, :]), dim=1)))
        
        return enc_output, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
    
    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim], 为 Decoder 的初始隐藏层状态，即论文中的 s_0
        # enc_output = [seq_len, batch_size, enc_hid_dim * 2]
        
        batch_size = enc_output.shape[1]
        seq_len = enc_output.shape[0]
        
        # repeat s seq_len times, 将 s 叠加 seq_len 次, 以便与所有 Encoder 隐藏层状态计算分数 (e_{ij})
        # s = [batch_size, seq_len, dec_hid_dim]
        # enc_output = [batch_size, seq_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, seq_len, 1)
        enc_output = enc_output.transpose(0, 1)
        
        # energy = [batch_size, seq_len, dec_hid_dim], 还不是 e_{ij} 构成的矩阵, 还要通过全连接层
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        
        # attention = [batch_size, seq_len]
        attention = self.v(energy).squeeze(2)  # 此时就是 e_{ij} 构成的矩阵
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        
        dec_input = dec_input.unsqueeze(1)  # dec_input = [batch_size, 1]
        
        embedded = self.dropout(self.embedding(dec_input)).transpose(0, 1)  # embedded = [1, batch_size, emb_dim]
        
        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)
        
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        enc_output = enc_output.transpose(0, 1)
        
        # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output).transpose(0, 1)
        
        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)
        
        # dec_output = [1, batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))
        
        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        
        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))
        
        return pred, dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder # Encoder()
        self.decoder = decoder # Decoder()
        self.device = device
    
    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """
        :param x: [seq_len, batch_size]
        :param y: [trg_len, batch_size]
        :param teacher_forcing_ratio: probability to use teacher forcing
        :return:
        """
        batch_size = x.shape[1]
        trg_len = y.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(x)
        
        # first input to the decoder is the <sos> tokens
        dec_input = y[0, :]
        
        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input, s, enc_output) # 逐步解码得到输出
            
            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = y[t] if teacher_force else top1
        
        return outputs

