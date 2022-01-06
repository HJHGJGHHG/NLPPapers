import torch
import torch.nn as nn
from torch.nn import functional as F


class SiamLSTM(nn.Module):
    @staticmethod
    def exponent_neg_manhattan_distance(left, right):
        pairwise_distance = torch.nn.PairwiseDistance(p=1)
        return torch.exp(-pairwise_distance(left, right))

    def __init__(self, max_seq_len, embeddings, args):
        super(SiamLSTM, self).__init__()

        self.max_seq_len = max_seq_len
        self.args = args
        self.embedding_layer = torch.nn.Embedding(len(embeddings), args.embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = False

        self.left_hidden_layer = torch.nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size, batch_first=True)
        self.right_hidden_layer = torch.nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size, batch_first=True)

    def forward(self, sentence_A, sentence_B, score=None):
        encoded_left_input = self.embedding_layer(sentence_A)
        encoded_right_input = self.embedding_layer(sentence_B)

        _, (left_h, _) = self.left_hidden_layer(encoded_left_input)
        _, (right_h, _) = self.right_hidden_layer(encoded_right_input)

        pred_score = SiamLSTM.exponent_neg_manhattan_distance(left_h[0], right_h[0])
        pred_score = torch.squeeze(pred_score)
        if score is not None:
            criterion = nn.MSELoss()
            loss = criterion(pred_score, score)
            return loss, pred_score

        return pred_score
