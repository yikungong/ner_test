# -*- coding: utf-8 -*-

import torch.nn as nn
from model.bilstm import BiLSTM
from model.crf import CRF
from utils.params import USE_CRF


class Models(nn.Module):
    def __init__(self, data, vocab_size, tag_size):
        super(Models, self).__init__()
        self.bilstm = BiLSTM(vocab_size, tag_size)
        self.crf = CRF(data.tag2id, tag_size)

    def forward(self, batch_inputs):
        if USE_CRF:
            inputs, batch_tag, seq_lens, mask = batch_inputs
            feats = self.bilstm.get_lstm_feature(inputs, seq_lens)  # 16, **, 12
            loss = self.crf(feats, batch_tag, mask).sum()
            tag_seq = self.crf.viterbi_decode(feats, mask)
        else:
            loss, tag_seq = self.bilstm.neg_log_likelihood_loss(batch_inputs)
        return loss, tag_seq

    def get_test_result(self, batch_inputs):
        if USE_CRF:
            inputs, batch_tag, seq_lens, mask = batch_inputs
            feats = self.bilstm.get_lstm_feature(inputs, seq_lens)  # 16, **, 12
            tag_seq = self.crf.viterbi_decode(feats, mask)
        else:
            tag_seq = self.bilstm(batch_inputs)
        return tag_seq
