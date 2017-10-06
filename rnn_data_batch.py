import torch
import numpy as np
import random

class Dataset(object):
    def __init__(self, sentences, sentences_len, targets):
        self.sentences_len = sentences_len
        self.torch_sentences = torch.from_numpy(np.array(sentences, dtype=np.float))
        self.torch_targets = torch.from_numpy(np.array(targets, dtype=np.float))

    def __call__(self, *args, **kwargs):
        return self.torch_sentences, self.sentences_len, self.torch_targets


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.sentences, self.sentences_len, self.targets = dataset()
        self.shuffle_packed_data(shuffle)
        self.max_sentences_num = int(len(self.sentences) / batch_size)
        self.batch_size = batch_size

    def __rnn_data_prepare(self, start, end):
        # sort and prepare the data for rnn pack_padded_sequence function
        inst = self.sentences[start:end].copy()
        inst_len = self.sentences_len[start:end].copy()
        targets = self.targets[start:end].copy()

        zipped = sorted(zip(inst, inst_len, targets), key=lambda attr: attr[1], reverse=True)
        sentence_batch, sentence_batch_len, sentence_targets = map(list, zip(*zipped))

        return torch.stack(sentence_batch, 0), sentence_batch_len, torch.from_numpy(np.array(sentence_targets))

    def shuffle_packed_data(self, shuffle):
        if shuffle:
            packed_data = list(zip(self.sentences, self.sentences_len, self.targets))
            random.shuffle(packed_data)
            self.sentences, self.sentences_len, self.targets = map(list, zip(*packed_data))

    def __iter__(self):
        list_b = []
        for sub_num in range(self.max_sentences_num + 1):
            start = self.batch_size * (sub_num)
            end = start + self.batch_size
            list_b.append(self.__rnn_data_prepare(start, end))
        print(list_b)
        return iter(list_b)

