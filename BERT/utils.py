import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import numpy as np
import pandas as pd


def dump_embedding():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    clinic_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    embedding_layer = clinic_bert.embeddings.word_embeddings
    extract_list = []
    word2id = pickle.load(open('data/word2index_high_0505.pkl', 'rb'))
    word_list = word2id.keys()
    print(word_list)
    word_ids_bert = tokenizer.convert_tokens_to_ids(word_list)
    word_ids_bert[0] = 0
    for idx in word_ids_bert:
        extract_list.append(embedding_layer(torch.tensor(idx)).detach().numpy())
    pickle.dump(np.array(extract_list), open('data/embedding_word.pkl', 'wb'))


def trunc(x):
    x = eval(x)
    x = x[:512]
    return x


word_freq = {}


def compute_word_freq(x):
    global word_freq
    word_list = eval(x)
    for word in word_list:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1


to_remove_words = {'admission', 'date', 'discharge', 'service', 'also', 'x', 'patient'}


def remove_meanless_words(x):
    global to_remove_words
    word_list = eval(x)
    ret = []
    for word in word_list:
        if word in to_remove_words:
            continue
        ret.append(word)
    return ret


def check_seq_len(clean_data=True):
    df = pd.read_csv("data/May6.csv")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    if clean_data:
        input_sequence_list = df['CLEAN_WORDS']
        input_data = tokenizer.batch_encode_plus([" ".join(eval(e)) for e in input_sequence_list])
    else:
        input_sequence_list = df['CLEANED TEXT']
        input_ids = []
        for e in input_sequence_list:
            try:
                input_ids.append(tokenizer.encode(e))
            except Exception as ex:
                print(ex)
                print(e)
    # input_ids = input_data['input_ids']
    max_seq_len = 0
    min_seq_len = 512
    for e in input_ids:
        max_seq_len = max(max_seq_len, len(e))
        min_seq_len = min(min_seq_len, len(e))
    print(max_seq_len, min_seq_len)
    len_list = []
    for i, e in enumerate(input_ids):
        len_list.append(len(e))
    print(np.median(np.array(len_list)))


if __name__ == '__main__':
    dump_embedding()

    df = pd.read_csv("data/May6.csv")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    input_sequence_list = df['CLEANED TEXT']
    input_data = tokenizer.batch_encode_plus([" ".join(eval(e)) for e in input_sequence_list])
    input_ids = input_data['input_ids']
    max_seq_len = 0
    min_seq_len = 512
    for e in input_ids:
        max_seq_len = max(max_seq_len, len(e))
        min_seq_len = min(min_seq_len, len(e))

    invalid_cnt = 0
    for i, e in enumerate(input_ids):
        if len(e) < 10:
            invalid_cnt += 1
            print(df['TEXT'][i])
            print("============================================")
    print(invalid_cnt)

    trunc_cnt = 0
    for i, e in enumerate(input_ids):
        if len(e) > 256:
            trunc_cnt += 1
    print(trunc_cnt)

    len_list = []
    for i, e in enumerate(input_ids):
        len_list.append(len(e))
    print(np.median(np.array(len_list)))
