import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
import torch
import numpy as np
class RNNAutoEcoder(nn.Module):
    def __init__(self, dic_vocab_num):
        super(RNNAutoEcoder, self).__init__()

        self.dic_vocab_num = dic_vocab_num

        self.word_embeddings = nn.Embedding(dic_vocab_num, 15)

        self.lstm_encoder = nn.LSTM(
            input_size=15,
            hidden_size=128,
            dropout=.25,
            batch_first=True
        )

        self.nn_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dic_vocab_num)
        )

        self.targets = nn.Sequential(
            nn.Linear(128, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def RNN_Encoder(self, sentences, sentences_len):
        print(sentences, sentences_len)
        word_embeds = self.word_embeddings(sentences)
        word_embeds_packed = pack_padded_sequence(word_embeds, sentences_len, batch_first=True)
        output, (h_state, c_state) = self.lstm_encoder(word_embeds_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        return output, h_state

    def Simple_NN_Decoder(self, x):
        output = self.nn_decoder(x)
        output = F.log_softmax(output)
        return output

    def classifier(self, x):
        x = self.targets(x)
        x = F.log_softmax(x)
        return x