# encoding=utf-8
import torch
import torch.nn as nn
from net_structure import RNNAutoEcoder
from torch.autograd import Variable
import random
import numpy as np
import codecs
import vocab_process
import torch.utils.data as Data


BATCH_SIZE = 16
torch.manual_seed(1234)

clean_data_url = 'dataset/clean_data_1.txt'
target_data_url = 'dataset/targets_1.txt'
load_clean_data = codecs.open(clean_data_url, 'r')
load_targets = codecs.open(target_data_url, 'r')

instances = load_clean_data.read().splitlines()
targets = load_targets.read().splitlines()

dictionary, dic_vocab_num = vocab_process.vocab_dict(instances, 'dict_BOW.json')
inst, inst_len = vocab_process.sentence_loader(instances, dictionary)

RnnAE = RNNAutoEcoder(dic_vocab_num)
RnnAE_optimizer = torch.optim.Adam(RnnAE.parameters(), lr=0.15)
loss_func = nn.NLLLoss()


import rnn_data_batch as RNN_Data
dataset = RNN_Data.Dataset(inst, inst_len, targets)
dataloader = RNN_Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

for epoch in range(10):
    training_result = []

    for sentences, sentences_len, targets in dataloader:
        x = y = Variable(sentences).long()
        class_t = Variable(targets).long()

        rnn_output, h_state = RnnAE.RNN_Encoder(x, sentences_len)
        result = RnnAE.classifier(h_state.squeeze(0))
        loss = loss_func(result, class_t)

        print('pred_class', torch.max(result, 1)[1].cpu().view(-1).data.numpy())
        print('real_class', class_t.data.numpy())

        for i in range(len(rnn_output)):
            out = RnnAE.Simple_NN_Decoder(rnn_output[i, :, :])
            # print(out.size(), y[i][:out.size()[0]])
            loss += loss_func(out, y[i][:out.size()[0]])
            print('pred', torch.max(out, 1)[1].cpu().view(-1).data.numpy())
            print('real', y[i].data.numpy())

        print(loss)
        print('==============================================')

        RnnAE_optimizer.zero_grad()
        loss.backward()
        RnnAE_optimizer.step()