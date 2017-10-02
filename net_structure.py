import torch.nn as nn
import torch.nn.functional as F

class RNNAutoEcoder(nn.Module):
    def __init__(self, dic_vocab_num):
        super(RNNAutoEcoder, self).__init__()

        self.dic_vocab_num = dic_vocab_num
        self.word_embeddings = nn.Embedding(dic_vocab_num, 10)

        self.lstm_encoder = nn.LSTM(
            input_size=10,
            hidden_size=15,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(15, dic_vocab_num)
        )

        self.targets = nn.Sequential(
            nn.Linear(15, 2)
        )

    def forward(self, sentences):
        word_embeds = self.word_embeddings(sentences)
        x, (h_state, c_state) = self.lstm_encoder(word_embeds.view(-1, 1, 10))
        x = self.decoder(x.view(-1, 15))
        x = F.log_softmax(x)

        return x.view(-1, self.dic_vocab_num), h_state

    def categrate(self, h_state):
        input = h_state[:,-1,:]
        output = self.targets(input.view(-1, 15))
        output = F.log_softmax(output)
        return output