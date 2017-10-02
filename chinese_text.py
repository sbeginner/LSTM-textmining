# encoding=utf-8
import torch
import torch.nn as nn
from net_structure import RNNAutoEcoder
from torch.autograd import Variable
import random
import numpy as np
import codecs
import vocab_process

clean_data_url = 'dataset/clean_data_1.txt'
target_data_url = 'dataset/targets_1.txt'
load_clean_data = codecs.open(clean_data_url, 'r')
load_targets = codecs.open(target_data_url, 'r')

instances = load_clean_data.read().splitlines()
targets = load_targets.read().splitlines()

dictionary, dic_vocab_num = vocab_process.vocab_dict(instances, 'dict_BOW.json')
inst = vocab_process.sentence_loader(instances, dictionary)

train_inst_tmp = []
for item in zip(inst, targets):
    train_inst_tmp.append(item)
random.shuffle(train_inst_tmp)
train_inst = train_inst_tmp[:100]
test_inst = train_inst_tmp[100:120]

RnnAE = RNNAutoEcoder(dic_vocab_num)
RnnAE_optimizer = torch.optim.Adam(RnnAE.parameters(), lr=0.15)
loss_func = nn.NLLLoss()

for epoch in range(1000):
    training_result = []
    testing_result = []
    for sentence in train_inst:
        train_x = train_y = Variable(torch.from_numpy(np.array(sentence[0])).long())
        class_t = Variable(torch.from_numpy(np.array([sentence[1]], dtype=np.float)).long())

        output, h_state = RnnAE.forward(train_x.view(1, -1))
        t_output = RnnAE.categrate(h_state)

        RnnAE_optimizer.zero_grad()

        loss = loss_func(output.view(-1, dic_vocab_num), train_y.view(-1))
        loss += loss_func(t_output.view(-1, 2), class_t)

        loss.backward()
        RnnAE_optimizer.step()

        training_result.append([torch.max(t_output.view(-1, 2), 1)[1].data.numpy(),
                                class_t.data.numpy()])
    print(sum(item[0] == item[1] for item in training_result), len(training_result))

    for sentence in test_inst:
        train_x = train_y = Variable(torch.from_numpy(np.array(sentence[0])).long())
        class_t = Variable(torch.from_numpy(np.array([sentence[1]], dtype=np.float)).long())

        output, h_state = RnnAE.forward(train_x.view(1, -1))
        t_output = RnnAE.categrate(h_state)

        testing_result.append([torch.max(t_output.view(-1, 2), 1)[1].data.numpy(),
                                class_t.data.numpy()])
    print(sum(item[0] == item[1] for item in testing_result), len(testing_result))

torch.save(RnnAE, 'RnnAE_trained')

