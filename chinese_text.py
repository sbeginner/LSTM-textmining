# encoding=utf-8
import torch
import torch.nn as nn
from net_structure import RNNAutoEcoder
from torch.autograd import Variable
import numpy as np
import codecs
import vocab_process

clean_data_url = 'dataset/clean_data_1.txt'
target_data_url = 'dataset/targets_1.txt'
TIME_STEP = 10
INPUT_SIZE = 10
OUTPUT_SIZE = 10
LR = .5
h_state = None

load_clean_data = codecs.open(clean_data_url, 'r')
load_targets = codecs.open(target_data_url, 'r')

instances = load_clean_data.read().splitlines()
targets = load_targets.read().splitlines()
targets_l = Variable(torch.from_numpy(np.array(targets, dtype=float)).long())

dictionary, dic_vocab_num = vocab_process.vocab_dict(instances, 'dict_BOW.json')
sentences_w2i = vocab_process.sentence_word2ind(instances, dictionary)

RnnAE = RNNAutoEcoder(dic_vocab_num)
RnnAE_optimizer = torch.optim.Adam(RnnAE.parameters(), lr=0.15)
loss_func = nn.NLLLoss()

for epoch in range(1000):
    count = 0
    loss_total = 0
    for index, sentence in sentences_w2i.items():

        train_x = train_y = Variable(torch.from_numpy(np.array(sentence)).long())

        output, h_state = RnnAE.forward(train_x.view(1, -1))
        t_output = RnnAE.categrate(h_state)

        RnnAE_optimizer.zero_grad()
        loss = loss_func(output.view(-1, dic_vocab_num), train_y.view(-1))
        loss += loss_func(t_output.view(-1, 2), targets_l[index])

        loss_total += loss

        loss.backward()
        RnnAE_optimizer.step()

        if torch.max(t_output.view(-1, 2), 1)[1].data.numpy() == targets_l[index].data.numpy():
            count += 1
        print(index)
        # print(sum(torch.max(output.view(-1, dic_vocab_num), 1)[1] == train_y.view(-1)).data.numpy()[0]/len(train_y.view(-1)))

    if epoch%100 == 0:
        print(loss_total)
        print(count / 10)
        # print(count/len(sentences_w2i))
