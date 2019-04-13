# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import utils.functions as uf
from utils.params import *
import torch.autograd as autograd

START_TAG = -2
STOP_TAG = -1

def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
                                                                                                                m_size)  # B * M


class CRF(nn.Module):
    def __init__(self, tag2id, tag_size):
        super(CRF, self).__init__()
        self.tag2id = tag2id
        self.tag_size = tag_size
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = torch.randn(tag_size, tag_size)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions[tag2id['<START>'], :] = -10000
        self.transitions[:, tag2id['<STOP>']] = -10000
        self.transitions[:, tag2id['<PAD>']] = -10000  # padding
        self.transitions[tag2id['<PAD>'], :] = -10000
        if USE_GPU:
            self.transitions = self.transitions.cuda()
        ''' j->i
         i/j    start     a       b       c      stop
       start    -10000  -10000  -10000  -10000  -10000
         a                                      -10000
         b                                      -10000
         c                                      -10000
        stop                                    -10000
        '''

    def calculate(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # print feats.view(seq_len, tag_size)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.next()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # print cur_partition.data

            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
            # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size,
                                                                         tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def _forward_alg(self, feats, mask):
        batch_size = feats.size(0)  # [B, L, C]
        max_len = feats.size(1)

        # Do the forward algorithm to compute the partition function
        # '<START>' has all of the score.
        # Wrap in a variable so that we will get automatic backprop
        forward_var = torch.full((batch_size, self.tag_size), -10000.)  # [B, C]
        forward_var[:, self.tag2id['<START>']] = 0.
        if USE_GPU:
            forward_var = forward_var.cuda()

        # [C, C] -> [1, C, C]
        transitions = self.transitions.unsqueeze(0)

        # Iterate through the sentence
        for idx in range(max_len):
            # broadcast the emission score: it is the same regardless of the previous tag
            emit_score = feats[:, idx].unsqueeze(2)  # [B, C, 1]

            # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
            # [B, 1, C] + [B, C, 1] + [1, C, C] -> [B, C, C]
            next_tag_var = forward_var.unsqueeze(1) + emit_score + transitions
            # [B, C, C] -> [B, C]
            next_tag_var = uf.log_sum_exp(next_tag_var)

            # padding part stay common, update word part
            mask_i = mask[:, idx].unsqueeze(1)  # [B, 1]
            forward_var = next_tag_var * mask_i + forward_var * (1 - mask_i)

        # [B, C] + [1, C] -> [B, C]
        terminal_var = forward_var + self.transitions[self.tag2id['<STOP>']]
        alpha = uf.log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, scores, tags, mask):
        """
         Gives the score of a provided tag sequence
        :param feats: [B, L, C]
        :param tags: [B, L]
        :param mask: [B, L]
        :return:
        """
        # batch_size = feats.size(0)  # [B, L, C]
        # max_len = feats.size(1)
        # score = torch.zeros(batch_size)
        # start_tag = torch.LongTensor([[self.tag2id['<START>']]] * batch_size)
        # if USE_GPU:
        #     score = score.cuda()
        #     start_tag = start_tag.cuda()
        # tags = torch.cat([start_tag, tags], -1)  # [B, L+1]
        # for idx in range(max_len):
        #     # feat[B,L,C]->[L,C]  t[B,C+1]->[C+1] -> [B]
        #     # zero-dimensional tensor cannot be concatenated
        #     idy = idx + 1
        #     emit_score = torch.cat([feat[idx, t[idy]] for feat, t in zip(feats.unsqueeze(3), tags)])
        #     # t[B,C+1] -> [B]
        #     trans_score = torch.cat([self.transitions[t[idy], t[idx]] for t in tags.unsqueeze(2)])
        #     score += (emit_score + trans_score) * mask[:, idx]
        # last_tag = tags.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        # score = score + self.transitions[self.tag2id['<STOP>'], last_tag]
        # return score

        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if USE_GPU:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        ## transition for label to STOP_TAG
        end_transition = self.transitions[:, STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len,
                                                                                         batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    #重点理解
    def viterbi_decode(self, feats, mask):
        batch_size = feats.size(0)  # [B, L, C]
        max_len = feats.size(1)

        # Initialize the viterbi variables in log space
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = torch.full((batch_size, self.tag_size), -10000.)
        forward_var[:, self.tag2id['<START>']] = 0
        bptr = torch.LongTensor()
        mask_r_cat = torch.zeros(batch_size, 1)
        bptr_cat = torch.zeros(batch_size, 1, self.tag_size).long()

        if USE_GPU:
            forward_var = forward_var.cuda()
            bptr = bptr.cuda()
            mask_r_cat = mask_r_cat.cuda()
            bptr_cat = bptr_cat.cuda()

        # [C, C] -> [1, C, C]
        transitions = self.transitions.data.unsqueeze(0)
        # [B, L] -> [B, 1+L]
        mask_r = torch.cat([mask_r_cat, (1 - mask.float())], 1).byte()

        pointers = []

        # Iterate through the sentence
        for idx in range(max_len):

            # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
            # [B, 1, C] + [1, C, C] -> [B, C, C]
            next_var = forward_var.unsqueeze(1) + transitions
            # best previous scores and tags
            # [B, C, C] -> [B, C]
            next_var, bptr_t = next_var.max(-1)
            # [B, C] + [B, C] -> [B, C]
            next_var += feats[:, idx]
            # [B, 0, C] -> [B, 1, C] -> ... -> [B, L-1, C]
            pointers.append(bptr_t)
            bptr = torch.cat([bptr, bptr_t.unsqueeze(1)], 1)

            # padding part stay common, update word part
            mask_i = mask[:, idx].unsqueeze(1)  # [B, 1]
            forward_var = next_var * mask_i.float() + forward_var * (1 - mask_i.float())

        # [B, C] + [1, C] -> [B, C]
        terminal_var = forward_var + self.transitions[self.tag2id['<STOP>']]
        best_score, best_tag = torch.max(terminal_var, -1)  # B

        # back-tracking - 还需要优化一下
        bptr.masked_fill_(mask_r[:, :-1].unsqueeze(2).expand(batch_size, max_len, self.tag_size), 0)
        bptr = torch.cat([bptr, bptr_cat], 1)
        bptr.scatter_(1, mask.sum(1).long().view(batch_size, 1, 1).expand(batch_size, 1, self.tag_size)
                      , best_tag.view(batch_size, 1, 1).expand(batch_size, 1, self.tag_size))

        best_path = torch.LongTensor(batch_size, max_len+1)
        if USE_GPU:
            best_path = best_path.cuda()
        best_tag = best_tag.unsqueeze(1)
        best_path.scatter_(1, mask.sum(1).long().unsqueeze(1), best_tag)
        for idx in range(max_len - 1, -1, -1):
            best_tag = torch.gather(bptr[:, idx], 1, best_tag)
            best_path[:, idx] = best_tag.view(1, -1)
        return best_path[:, 1:]

        # 第二种方案，循环多，很慢
        # bptr1 = bptr1.tolist()
        # best_path = best_tag1.unsqueeze(1).tolist()
        # for b in range(batch_size):
        #     x = best_tag1[b]  # best tag
        #     y = int(mask[b].sum().item())
        #     for bptr_t in reversed(bptr1[b][:y]):
        #         x = bptr_t[x]
        #         best_path[b].append(x)
        #     start = best_path[b].pop()
        #     assert start == 1  # Sanity check
        #     best_path[b].reverse()
        # for i, ai in enumerate(a):
        #     print(ai)
        #     print(best_path[i])
        #     print(operator.eq(ai[1:len(best_path[i])+1], best_path[i]))
        #     # best_path[b] += [0] * (max_len - len(best_path[b]))
        # best_path = torch.Tensor(best_path)
        # if USE_GPU:
        #     best_path = best_path.cuda()
        # return best_path

    def forward(self, feats, tags, mask):
        #forward_score = self._forward_alg(feats, mask)  # 1,16
        forward_score, scores = self.calculate(feats, mask)
        gold_score = self._score_sentence(scores, tags, mask)
        return forward_score - gold_score


def test_decode(feats, trans, mask):
    batch_size = feats.size(0)  # [B, L, C]
    max_len = feats.size(1)

    # Initialize the viterbi variables in log space
    init_vvars = torch.full((batch_size, 4), -10000.) # 2,4
    init_vvars[:, 1] = 0

    # forward_var at step i holds the viterbi variables for step i-1
    forward_var = init_vvars
    print(forward_var)

    bptr = torch.LongTensor()

    # [C, C] -> [1, C, C]
    transitions = trans.data.unsqueeze(0)
    print(transitions)

    # Iterate through the sentence
    for idx in range(max_len):

        # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
        # [B, 1, C] + [1, C, C] -> [B, C, C]
        next_var = forward_var.unsqueeze(1) + transitions
        print(next_var)
        # best previous scores and tags
        # [B, C, C] -> [B, C]
        next_var, temp_id = next_var.max(-1)
        print(next_var)
        print(temp_id)
        # [B, C] + [B, C] -> [B, C]
        next_var += feats[:, idx]
        print(feats)
        print(feats[:, idx])
        print(next_var)
        # [B, 1, C] -> [B, 2, C] -> ... -> [B, L, C]
        print(temp_id.unsqueeze(1)) # 2,1,4
        bptr = torch.cat((bptr, temp_id.unsqueeze(1)), 1)
        print(bptr)

        # padding part stay common, update word part
        mask_i = mask[:, idx].unsqueeze(1)  # [B, 1]
        forward_var = next_var * mask_i + forward_var * (1 - mask_i)

    # [B, C] + [1, C] -> [B, C]
    terminal_var = forward_var + trans[2]
    best_score, best_tag = torch.max(terminal_var, -1)  # B

    # back-tracking
    bptr = bptr.tolist()
    best_path = best_tag.unsqueeze(1).tolist()
    print(best_path)
    for b in range(batch_size):
        x = best_tag[b]  # best tag
        y = int(mask[b].sum().item())
        for bptr_t in reversed(bptr[b][:y]):
            x = bptr_t[x]
            best_path[b].append(x)
        start = best_path[b].pop()
        best_path[b].reverse()
        assert start == 1  # Sanity check
    return best_score, best_path


if __name__ == '__main__':
    feats = torch.randn(2, 5, 4)
    trans = torch.randn(4, 4)
    trans[1, :] = -10000.
    trans[:, 2] = -10000.
    trans[0, :] = -10000.
    trans[:, 0] = -10000.
    tags = torch.LongTensor([[1, 2, 3, 0, 0], [2, 3, 1, 0, 0]])
    mask = torch.FloatTensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    test_decode(feats, trans, mask)

'''
a
tensor([[[1., 2., 3.]],
        [[4., 5., 6.]]])
b
tensor([[[1.],
         [2.],
         [1.]],
        [[2.],
         [3.],
         [1.]]])
a+b
tensor([[[2., 3., 4.],
         [3., 4., 5.],
         [2., 3., 4.]],
        [[6., 7., 8.],
         [7., 8., 9.],
         [5., 6., 7.]]])
'''
