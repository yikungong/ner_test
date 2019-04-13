# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.params import *


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(BiLSTM, self).__init__()
        self.bi_flag = 2 if BI_DIRECT else 1

        # 遇到index = 0用0填充
        self.char_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)

        self.char_drop = nn.Dropout(DROPOUT)

        # 双向LSTM，也就是序列从左往右算一次，从右往左又算一次，这样就可以两倍的输出 bidirectional
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM // self.bi_flag, batch_first=True,
                            num_layers=LSTM_LAYER, bidirectional=BI_DIRECT)

        self.hidden2tag = nn.Linear(HIDDEN_DIM, tag_size, bias=False)

        self.hidden = self.init_hidden()

        if USE_GPU:
            self.char_embedding = self.char_embedding.cuda()
            self.char_drop = self.char_drop.cuda()
            self.lstm = self.lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

    def init_hidden(self):
        return (torch.zeros(LSTM_LAYER * self.bi_flag, BATCH_SIZE, HIDDEN_DIM // self.bi_flag),
                torch.zeros(LSTM_LAYER * self.bi_flag, BATCH_SIZE, HIDDEN_DIM // self.bi_flag))

    def init_weight(self):
        nn.init.xavier_uniform(self.lstm.all_weights)

    def get_lstm_feature(self, inputs, seq_lengths):
        char_embeds = self.char_drop(self.char_embedding(inputs))
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        lstm_out, lstm_hidden = self.lstm(pack_input, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood_loss(self, batch_inputs):
        inputs, batch_tag, seq_lens, _ = batch_inputs
        batch_size = inputs.size(0)
        max_len = inputs.size(1)

        outs = self.get_lstm_feature(inputs, seq_lens).view(batch_size * max_len, -1)
        score = F.log_softmax(outs, 1)

        loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')
        total_loss = loss_function(score, batch_tag.view(batch_size * max_len))

        _, tag_seq = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, max_len)

        return total_loss, tag_seq

    def forward(self, batch_inputs):
        inputs, _, seq_lens, _ = batch_inputs
        outs = self.get_lstm_feature(inputs, seq_lens)
        _, tag_seq = torch.max(outs, -1)
        return tag_seq
