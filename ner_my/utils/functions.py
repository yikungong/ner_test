# -*- coding: utf-8 -*-

import torch
import pickle
from utils.params import *


class Data(object):
    def __init__(self):
        self.char2id = {'<UNK>': 0}
        self.id2char = {0: '<UNK>'}
        self.tag2id = {'<PAD>': 0, '<START>': 1, '<STOP>': 2}
        self.id2tag = {0: '<PAD>', 1: '<START>', 2: '<STOP>'}
        self.train_infos = []
        self.dev_infos = []
        self.test_info = []

    def create_vocab(self, input_file):
        """
        Get the vocabulary and tag , convert them to id
        :param input_file:
        :return:
        """
        print("create the vocabulary and tag , convert them to id ...")

        sents_info = []

        with open(input_file, 'r') as f:
            sent, tag = [], []
            for i, line in enumerate(f.readlines()):
                if i % 1000 == 0:
                    print(i, line)
                if len(line.strip()) > 2:
                    ls = line.strip().split()
                    c = ls[0]
                    t = ls[-1]
                    if c not in self.char2id.keys():
                        self.char2id[c] = len(self.char2id)
                        self.id2char[len(self.id2char)] = c

                    if t not in self.tag2id.keys():
                        self.tag2id[t] = len(self.tag2id)
                        self.id2tag[len(self.id2tag)] = t

                    sent.append(c)
                    tag.append(t)
                else:
                    assert len(sent) == len(tag)
                    sent_id = [self.char2id.get(c, 0) for c in sent]
                    tag_id = [self.tag2id.get(t, 0) for t in tag]
                    sents_info.append([sent, tag, sent_id, tag_id, len(sent)])
                    sent, tag = [], []
        return sents_info

    def set_infos(self, input_file, type):
        if type == 'train':
            self.train_infos = self.create_vocab(input_file)
        elif type == 'dev':
            self.dev_infos = self.create_vocab(input_file)

    def load(self, data_file):
        with open(data_file, "rb") as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def save(self, input_file):
        with open(input_file, "wb") as f:
            pickle.dump(self.__dict__, f, 2)


def generate_batch_data(data_infos, batch_size):
    """
    生成批量数据
    :param data_infos: [sent, tag, sent_id, tag_id, len_seq]
    :param batch_size:
    :return:
    """

    sents = [s[2] for s in data_infos]
    tags = [s[3] for s in data_infos]
    seq_lens = [s[4] for s in data_infos]
    total_batch = len(data_infos) // batch_size + 1

    for i in range(total_batch):

        start, end = i * batch_size, min((i + 1) * batch_size, len(data_infos))
        if start == end:
            continue
        max_len = max(seq_lens[start:end])

        len_batch, word_perm_idx = torch.tensor(seq_lens[start:end]).sort(0, descending=True)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)

        sent_batch = torch.tensor([line + [0.] * (max_len - len(line)) for line in sents[start:end]],
                                  requires_grad=True)[word_perm_idx].long()

        tag_batch = torch.tensor([line + [0.] * (max_len - len(line)) for line in tags[start:end]],
                                 requires_grad=True)[word_perm_idx].long()

        mask_batch = torch.gt(sent_batch, 0)

        if USE_GPU:
            sent_batch = sent_batch.cuda()
            tag_batch = tag_batch.cuda()
            len_batch = len_batch.cuda()
            word_seq_recover = word_seq_recover.cuda()
            mask_batch = mask_batch.cuda()

        yield sent_batch, tag_batch, len_batch, mask_batch


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    m, _ = torch.max(vec, -1)  # B,C
    return m + torch.log(torch.sum(torch.exp(vec - m.unsqueeze(-1)), -1))


if __name__ == '__main__':
    a = torch.randn(2, 3, 3)
    print(a)
    max_score, _ = torch.max(a, -1)
    print(max_score.unsqueeze(-1))
    print(a - max_score.unsqueeze(-1))
    max_score_broadcast = max_score.unsqueeze(-1)
    max_score_broadcast = max_score_broadcast.expand(a.size())
    print(a - max_score_broadcast)
    print(torch.sum(a - max_score_broadcast, -1))
    print(torch.log(torch.abs(torch.sum(a - max_score_broadcast, -1))))



