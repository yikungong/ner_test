# -*- coding: utf-8 -*-

import time
import torch
import random
import os
import torch.optim as optim
from utils import functions as uf
from model.models import Models
from utils import metric
from utils.params import *


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print("Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def prepared_data():

    data = uf.Data()

    data.set_infos(TRAIN_DIR, 'train')
    data.set_infos(DEV_DIR, 'dev')

    data.save(PICKLE_DIR)


def train():

    data = uf.Data()
    data.set_infos(TRAIN_DIR, 'train')
    data.set_infos(DEV_DIR, 'dev')
    # data.load(PICKLE_DIR)

    VOCAB_SIZE = len(data.char2id)
    TAG_SIZE = len(data.tag2id)

    random.shuffle(data.train_infos)

    print("Training model...")
    # model = BiLSTM(params)
    model = Models(data, VOCAB_SIZE, TAG_SIZE)
    if USE_GPU:
        model = model.cuda()
    print("model:{}".format(model))

    optimizer = None
    if OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2)
    elif OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2)
    else:
        print("Optimizer illegal: %s , use sgd or adam." % OPTIMIZER)
        exit(0)

    # start training
    best_dev, best_dev_epoch = -1, 1

    for idx in range(EPOCHS):

        print("Epoch: %s/%s" % (idx + 1, EPOCHS))
        if OPTIMIZER.lower() == "sgd":
            optimizer = lr_decay(optimizer, idx, LR_DECAY, LEARNING_RATE)

        # set model in train mode
        model.train()
        model.zero_grad()

        epoch_start = time.time()
        temp_start = epoch_start

        batch_id = 1
        total_loss = 0
        sample_loss = 0
        right_token = 0
        whole_token = 0

        for input_batch in uf.generate_batch_data(data.train_infos, BATCH_SIZE):
            batch_id += 1

            loss, tag_pred = model(input_batch)
            total_loss += loss
            sample_loss += loss

            _, tag_batch, len_batch, _ = input_batch
            right, whole = metric.predict_check(tag_pred, tag_batch, len_batch)
            right_token += right
            whole_token += whole

            if batch_id % 100 == 0:
                temp_cost = time.time() - temp_start
                temp_start = time.time()
                print("     Instance: %s*%s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    batch_id, BATCH_SIZE, temp_cost, sample_loss, right_token, whole_token,
                    (right_token + 0.) / whole_token))
                sample_loss = 0

            loss.backward()
            optimizer.step()
            model.zero_grad()

        temp_cost = time.time() - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
            batch_id, temp_cost, sample_loss, right_token, whole_token,
            (right_token + 0.) / whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, total loss: %s" % (idx + 1, epoch_cost, total_loss))

        # dev
        acc, p, r, f, _ = metric.evaluate(model, data.dev_infos, data.id2tag)
        dev_cost = time.time() - epoch_finish
        print("Dev: time: %.2fs; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, acc, p, r, f))

        if f > best_dev:
            best_dev = f
            best_dev_epoch = idx + 1
        print("the best f score:", best_dev)
        print("the best epoch:", best_dev_epoch)


if __name__ == '__main__':

    # 35935
    # print('gpu device:', torch.cuda.current_device())
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if not SEED_NUM:
        SEED_NUM = random.randint(1, 100000)
    random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)
    print('Random Seed num:', SEED_NUM)

    # prepared_data()

    train()
