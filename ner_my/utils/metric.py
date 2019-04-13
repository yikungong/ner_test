# -*- coding: utf-8 -*-
import utils.functions as uf
import numpy as np
from utils.params import *


def predict_check(pred_variable, gold_variable, len_variable):
    """
    训练阶段，计算accuracy
    :param pred_variable:
    :param gold_variable:
    :param len_variable:
    :return:
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()

    mask = np.zeros(pred_variable.size())
    for i, seq_len in enumerate(len_variable):
        mask[i, :seq_len] = 1

    flag = (pred == gold)
    right_token = np.sum(flag * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, len_variable, id2tag, word_recover):
    """
        将预测的id恢复为label
        input:
            pred_variable (batch_size, sent_len): pred tag result tag_seq
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    seq_lens = len_variable.cpu().data.numpy()

    batch_size = gold_variable.size(0)

    pred_label = []
    gold_label = []
    for i in range(batch_size):
        pred = [id2tag[pred_tag[i][j]] for j in range(seq_lens[i])]
        gold = [id2tag[gold_tag[i][j]] for j in range(seq_lens[i])]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def evaluate(model, data_infos, id2tag):
    """
    验证集 测试集 评估
    :param model:
    :param data_infos:
    :param id2tag:
    :return:
    """

    pred_results = []  # total pred result
    gold_results = []  # total gold result

    # set model in eval model
    model.eval()

    for input_batch in uf.generate_batch_data(data_infos, BATCH_SIZE):

        tag_pred = model.get_test_result(input_batch)

        _, tag_batch, len_batch, word_recover = input_batch
        pred_label, gold_label = recover_label(tag_pred, tag_batch, len_batch, id2tag, word_recover)
        pred_results += pred_label
        gold_results += gold_label

    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, TAG_SCHEME)
    # acc, p, r, f = get_ner_f(gold_results, pred_results)
    return acc, p, r, f, pred_results


def get_ner_f(golden_lists, predict_lists):
    """
    只有lstm层，为单个字打tag的f值
    :param golden_lists:
    :param predict_lists:
    :return:
    """
    right_info = {}
    predict_info = {}
    golden_info = {}

    for idx, (golden_list, predict_list) in enumerate(zip(golden_lists, predict_lists)):
        for golden, predict in zip(golden_list, predict_list):
            if golden not in right_info:
                right_info[golden] = 0
            if golden == predict:
                right_info[golden] = right_info[golden] + 1

        for key in set(golden_list):

            value = golden_info[key] if key in golden_info else 0
            golden_info[key] = golden_list.count(key) + value

            value = predict_info[key] if key in predict_info else 0
            predict_info[key] = predict_list.count(key) + value

    right_num = 0
    right_tag = 0

    predict_num = 0
    golden_num = 0
    all_tag = 0

    # entity info
    for k, v in golden_info.items():

        right_tag += right_info[k]
        all_tag += golden_info[k]

        if k != 'O':
            right_num += right_info[k]
            predict_num += predict_info[k]
            golden_num += v
            _, p, r, f = get_fmeasure(right_info[k], v, predict_info[k], 1, 1)
            print("label = %s, gold_num = %d, pred_num = %d, right_num = %d, p = %.3f, r = %.3f, f = %.3f" %
                  (k, v, predict_info[k], right_info[k], p, r, f))

    return get_fmeasure(right_num, golden_num, predict_num, right_tag, all_tag)


def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    """
    输入所有句子 label 的预测和真实值，计算f1
    :param golden_lists: 
    :param predict_lists: 
    :param label_type: 
    :return: 
    """
    
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0

    for idx, (golden_list, predict_list) in enumerate(zip(golden_lists, predict_lists)):
        for golden_tag, predict_tag in zip(golden_list, predict_list):
            if golden_tag == predict_tag:
                right_tag += 1
        all_tag += len(golden_list)

        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)

        # 交集
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)

    # print entity info
    get_entity_fmeasure([golden_full, predict_full, right_full])

    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return get_fmeasure(right_num, golden_num, predict_num, right_tag, all_tag)


def get_fmeasure(right_num, golden_num, predict_num, right_tag, all_tag):
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag
    return accuracy, precision, recall, f_measure


def get_entity_fmeasure(full_list):
    """
    统计各个实体信息
    :param full_list: golden_full, predict_full, right_full
    :return:
    """
    ner_counts = []
    for full in full_list:
        entity = list(map(lambda x: x.split(']')[1], full))
        ner_count = {w: entity.count(w) for w in set(entity)}
        ner_counts.append(ner_count)

    precisions, recalls, f_measures = {}, {}, {}
    for key, golden_num in ner_counts[0].items():

        precision, recall, f_measure = -1, -1, -1
        pred_num = ner_counts[1][key] if key in ner_counts[1] else 0
        right_num = ner_counts[2][key] if key in ner_counts[2] else 0
        if pred_num != 0:
            precision = (right_num + 0.0) / pred_num
        recall = (right_num + 0.0) / golden_num
        if not ((precision == -1) or (recall == -1) or (precision + recall) <= 0.):
            f_measure = 2 * precision * recall / (precision + recall)

        precisions[key] = precision
        recalls[key] = recall
        f_measures[key] = f_measure
        print("label = %s, gold_num = %d, pred_num = %d, right_num = %d, p = %.3f, r = %.3f, f = %.3f" %
              (key, golden_num, pred_num, right_num, precision, recall, f_measure))
    return precisions, recalls, f_measures


def reverse_style(input_string):

    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    list_len = len(label_list)
    begin_label = 'B'
    end_label = 'E'
    single_label = 'S'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        tags = current_label.split('-')
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = tags[-1] + '[' + str(i)
            index_tag = tags[-1]

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = tags[-1] + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


if __name__ == '__main__':
    print(get_ner_BIO(['O', 'O', 'B-PER', 'I-PER', 'I-FILM', 'O', 'O', 'I-ORG', 'O', 'I-ORG', 'B-ORG', 'I-POS', 'B-MUSIC']))
