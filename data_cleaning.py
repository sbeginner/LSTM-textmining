# encoding=utf-8
import jieba
import jieba.analyse
import codecs

raw_data_url = 'dataset/raw_data.txt'
clean_data_url = 'dataset/clean_data.txt'

def input_raw_data(url):
    raw_data = codecs.open(url, 'r')
    origin_data = raw_data.readlines()
    raw_data.flush()
    raw_data.close()
    return origin_data


def clean_processing(line):
    seg_list = jieba.cut(line, cut_all=False)
    origin = [item for item in seg_list if item]
    clean_data = ' '.join(origin)
    return clean_data


def output_clean_data(clean_data_set):
    clean_data = codecs.open(clean_data_url, 'w')
    clean_data.writelines(clean_data_set)
    clean_data.flush()
    clean_data.close()


origin_data = input_raw_data(raw_data_url)
clean_data_set = [clean_processing(line) for line in origin_data]
output_clean_data(clean_data_set)
