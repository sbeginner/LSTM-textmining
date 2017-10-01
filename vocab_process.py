import codecs
import json
import numpy as np
from pathlib import Path


# BOW-dict
def vocab_dict(instances, dict_url):
    # create or update the BOW dictionary (BOW-dict)
    if Path(dict_url).exists():
        dict_bow = codecs.open(dict_url, 'r+')
        dictionary = json.load(dict_bow)
        exist_vocab = dictionary.keys() | set()
    else:
        dict_bow = codecs.open(dict_url, 'w')
        dictionary = dict()
        exist_vocab = set()
    # new vocab is the word that can't find out in the exist BOW-dict
    new_vocab = exist_vocab ^ new_instances_vocab(instances)
    # update the BOW-dict
    dictionary = combine_vocab(new_vocab, exist_vocab, dictionary, dict_bow)

    return dictionary, len(dictionary)

def new_instances_vocab(instances):
    vocab = set()
    for instance in instances:
        vocab.update(instance.split())
    return vocab

def combine_vocab(new_vocab, exist_vocab, dictionary, dict_bow):
    if new_vocab:
        vocab2Ind = {word: index + len(exist_vocab) for index, word in enumerate(new_vocab)}
        dictionary.update(vocab2Ind)
        dict_bow.seek(0)
        dict_bow.truncate()
        json.dump(dictionary, dict_bow)
    dict_bow.close()
    return dictionary


# sentence: character(char) to index(int)
def sentence_word2ind(instances, dictionary):
    sentence_ind = []
    for instance in instances:
        sentence = [dictionary[word] if word in dictionary else -1 for word in instance.split()]
        sentence_ind.append(sentence)

    return dict(enumerate(np.array(sentence_ind)))