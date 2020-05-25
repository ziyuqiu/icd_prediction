# main.py:
# model, model_save_fname, dev_preds, preds_ids = main(
#         train_file="exps-data/data/train_data.pkl",     # document, one hot labels (dim = #codes), doc id
#         dev_file="exps-data/data/dev_data.pkl",
#         lang="en",
#         load_pretrain_ft=True,                          # fast text embeddings (don't have, need to change to BioBert)
#         load_pretrain_pubmed=False,
#         pretrain_file="../cc.en.300.vec",
#         model_name="slstm",
#         device=device
#     )

#     load_data.py:
#     def get_titles_T # title index matrix, title vocab 
#     def load_ft_embeds # need to change to BioBert, or just don't use and train from scratch

#     train.py:
#     eval_data = (logits, preds, labels, ids, avg_loss) # preds: dim = B x C, probs



# generate_preds_file(
#         dev_preds,                                          # N x C
#         preds_ids,                                          # doc ids (maybe bc shuffled)
#         mlb_file="exps-data/data/mlb.pkl",                  # should be multilingual bert pretrained embeddings 
#         devids_file="exps-data/data/ids_development.txt",   # doc ids
#         preds_file="./preds_development.txt"                # output, format: doc id, code | code | code ... 
#     )

#     preds[i, :].astype(bool) # should be probs -> 0/1 as a mask to look for class in mlb
#                              # mlb = sklearn.preprocessing.MultiLabelBinarizer, mlb.classes_ = list of codes
#                              # so the order of labels in mlb should be the same as binary labels
#                              # details in read_data.py
#                              # train/dev data已经有了, only need to fit mlb from (train+dev) labels, then transform labels in to binary



# eval_cmd = 'python evaluation.py --ids_file="{}" --anns_file="{}" --dev_file="{}" --out_file="{}"'
# eval_cmd = eval_cmd.format(
#     "exps-data/data/ids_development.txt",
#     "exps-data/data/anns_train_dev.txt",                    # should be answer, format: doc id, code | code | code ... 
#     "preds_development.txt",                                # output, format: doc id, code | code | code ... 
#     "eval_output.txt"                                       # those in data/results, format: TP/FP/FN, doc id, code (one doc multiple lines)
# )

# use predict.py for summary Results.txt


'''
TODO 0423
__main__:
main > get_data:
needs: train data, dev data file: doc, doc id from SAMPLE10K.csv (split)
produces: word2index (already got)
mlb: get together with binary label in data (read data里有用的只有mlb那里)
ids_development
main > get_data > get_X_y_ids: 
word2index > X, y, ids tensor
main > dataloader: sample X, y, ids with batched_data
main > models: clstm need title_vocab, embed_matrix_T, could be produced with D_ICD_DIAGNOSES.csv and diag2idx
'''


import pickle as pkl
import pandas as pd 
from collections import Counter
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from pprint import pprint
import pandas as pd 

'''generate codes_and_titles'''
def codes_and_titles(codetitle_file, code2index_file):
    print('===== start generating {}, {} ====='.format(codetitle_file, code2index_file))
    with open(data_path + 'diag2idx.pickle', "rb") as rf:
        diag2idx = pkl.load(rf)
    codes_in_dict = set(diag2idx.keys())

    code_set = set()
    codes = sample['DIAG_CODES'].values.tolist()
    for c1 in codes:
        for c2 in c1.split(','):
            code_set.add(c2)
    print('number of codes in {}:{}'.format(sample_file, len(code_set)))

    code_set = code_set.intersection(codes_in_dict)
    print('number of codes in dict: {}'.format(len(code_set)))
    # print(code_set)

    selected = []
    codes = list(code_set)
    codes.sort()
    # code2index = {"<pad>":0, "<unk>": 1}
    code2index = {}
    df = pd.read_csv(data_path + 'D_ICD_DIAGNOSES.csv')
    for i, code in enumerate(codes):
        code2index[code] = i
        title = df.loc[df['ICD9_CODE'] == code, 'SHORT_TITLE'].values
        if len(title) == 0: continue
        title = title[0]
        title = title.replace(',', '')
        title = title.replace('"', '')
        selected.append(title)
    selected = pd.DataFrame(list(zip(codes, selected)), columns =['code', 'title']) 
    selected.to_csv(out_path + codetitle_file, index=False)
    print('codes and titles out to: {}'.format(codetitle_file))

    with open(out_path + code2index_file, 'wb') as wf:
        pkl.dump(code2index, wf)
    print('len(code2index): {}'.format(len(code2index)))
    print(code2index)
    print('code2index out to: {}'.format(code2index_file))

    return code2index


'''generate title tensor and title2index'''
def build_vocab(texts, min_df=5, max_df=0.6, keep_n=10000):
    counter = Counter([token for text in texts for token in text.split()])
    counter = Counter({k:v for k, v in counter.items() if min_df <= v <= int(len(texts)*max_df)})
    words = [w for w, _ in counter.most_common()[:keep_n-2]]
    word2index = {"<pad>":0, "<unk>": 1}
    for i in range(len(words)):
        word2index[words[i]] = i+2
    return word2index

def text_to_seq(text, word2index):
    return [word2index[token] if token in word2index else word2index["<unk>"] for token in text.split()]

def pad_seq(seq, max_len):
    seq = seq[:max_len]
    seq += [0 for i in range(max_len - len(seq))]
    return seq

def get_titles_T(codes_titles_file):
    titles = []
    with open(codes_titles_file, "r") as rf:
        for line in rf:
            if line == 'ICD9_CODE,SHORT_TITLE': continue
            title, code = line.split(",")
            titles.append(title)
    titles_vocab = build_vocab(titles, 1, 1.0)
    titles = [pad_seq(text_to_seq(i, titles_vocab), 10) for i in titles]
    titles = torch.tensor(titles).long()
    return titles, titles_vocab

def title_tensor_vocab(code_title_file, title2index_file, title2tensor_file):
    print('===== start generating {}, {} ====='.format(title2index_file, title2tensor_file))
    titles, title_vocab = get_titles_T(out_path + code_title_file)
    print('title tensor shape: {}'.format(titles.shape))
    # print(title_vocab) 
    print('title2index: {}'.format(len(title_vocab))) 

    with open(out_path + title2index_file, "wb") as wf:
        pkl.dump(title_vocab, wf)
    with open(out_path + title2tensor_file, "wb") as wf:
        pkl.dump(titles, wf)
    print('title2idnex out to: {}'.format(title2index_file))
    print('title2tensor out to: {}'.format(title2tensor_file))


'''generate mlb, labels, anns file '''
# mlb.classes_ should be in the same order as code2index
# keep one should be enough but too lazy
def mlb(mlb_file, anns_file):
    print('===== start generating {}, {} ====='.format(mlb_file, anns_file))
    ids = sample['Unnamed: 0'].values.tolist()
    labels = sample['DIAG_CODES'].values.tolist()
    for i in range(len(labels)): 
        label_set = set()
        for t in labels[i].split(','):
            # if t in code2index:
            label_set.add(t) 
        label_list = list(label_set)
        label_list.sort()
        labels[i] = label_list
        # labels[i] = [t for t in labels[i].split(',') if t in code2index]

    anns = []
    for i in range(len(labels)):
        anns.append(str(ids[i]) + '\t' + '|'.join(labels[i]) + '\n')
    with open(out_path + anns_file, 'w') as wf:
        for i in range(len(anns)):
            wf.write(anns[i])
    print('len(anns):', len(anns))
    print('anns out to: {}'.format(anns_file))

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    print('labels shape: {}'.format(labels.shape))
    print('number of classes: {}'.format(len(mlb.classes_)))
    print('mlb.classes_:', mlb.classes_)

    with open(out_path + mlb_file, 'wb') as wf:
        pkl.dump(mlb, wf)
    print('mlb out to: {}'.format(mlb_file))

    return mlb, labels, anns


'''train data: tup(document, one hot labels (dim = #codes), doc id)'''
# actually it's all data, split afterwards
def process_data_vocab(data_file, word2index_file):
    print('===== start generating {}, {} ====='.format(data_file, word2index_file))
    train_data = []
    text = []
    for i, row in sample.iterrows():
        doc_id = row[0]
        doc = row['STEM_WORDS'][1:-1]
        words = doc.split(', ')
        words = [w[1:-1] for w in words]
        text.extend(words)
        doc = ' '.join(words)
        label = labels[i, :]
        train_data.append((doc, label, doc_id))
    # print(train_data[0])

    with open(out_path + data_file, 'wb') as wf:
        pkl.dump(train_data, wf)
    print('processed train data out to: {}'.format(data_file))

    # with open(path + 'word2idx.pickle', "rb") as rf:
    #     word2index = pkl.load(rf)
    # # print(word2index) # '<OTHER>': 0
    # del word2index['<OTHER>']
    # del word2index['pt']
    # word2index['<pad>'] = 0
    # word2index['<unk>'] = 1 # 应该可以直接把word1去掉, 最常见的变成unk, could also just rebuild vocab

    word2index = build_vocab(text, 1, 1.0)
    print('vocab size: {}'.format(len(word2index)))
    print(list(word2index.items())[:5])
    with open(out_path + word2index_file, "wb") as wf:
        pkl.dump(word2index, wf)
    print('word2index out to: {}'.format(word2index_file))

    return train_data, word2index


'''data to tensor, generate equivalence to ids_development.txt'''
def get_X_y_ids(data, word2index, max_seq_len=256):
    X, y, doc_ids = [], [], []
    for idx, val in enumerate(data):
        text, labels, doc_id = val
        X.append(pad_seq(text_to_seq(text, word2index), max_seq_len))
        y.append(labels)
        doc_ids.append(doc_id)
    
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float)
    doc_ids_tensor = torch.tensor(doc_ids, dtype=torch.long)
    X = X.view(-1, max_seq_len)
    
    num_classes = len(y[0])
    y = y.view(-1, num_classes)
    doc_ids_tensor = doc_ids_tensor.view(-1)
    
    return X, y, doc_ids_tensor, doc_ids

def data_tensor(data, word2index, data_file, ids_file):
    print('===== start generating {}, {} ====='.format(data_file, ids_file))
    X, y, ids_tensor, ids_list = get_X_y_ids(data, word2index)
    print(X.shape, y.shape, ids_tensor.shape) 
    data = (X, y, ids_tensor)
    with open(out_path + data_file, "wb") as wf:
        pkl.dump(data, wf)
    print('(X, y, ids) data tensors out to: {}'.format(data_file))

    with open(out_path + ids_file, 'w') as wf:
        for i in ids_list:
            wf.write(str(i) + '\n')
    print('ids out to: {}'.format(ids_file))

    return X, y, ids_tensor, ids_list


'''create ids, anns file'''
# def ids_anns(ids_file, anns_file):
#     print('===== start generating {}, {} ====='.format(ids_file, anns_file))
#     ids = sample['Unnamed: 0'].values.tolist()
#     with open(out_path + ids_file, 'w') as wf:
#         for i in ids:
#             wf.write(str(i) + '\n')
#     print('ids out to: {}'.format(ids_file))
#     codes = sample['DIAG_CODES'].values.tolist()
#     codes = ['|'.join(c.split(',')) for c in codes]
#     assert len(ids) == len(codes)
#     with open(out_path + anns_file, 'w') as wf:
#         for i in range(len(ids)):
#             wf.write(str(ids[i]) + '\t' + codes[i] + '\n')
#     print('anns out to: {}'.format(anns_file))

'''for the easy of main input, split tensors, test ids, test anns'''
def split(X, y, ids_tensor, ids_list, anns, dev_id, test_id, tensor_files, txt_files):
    print('===== start generating splits =====')
    Xtrain, Xdev, Xtest = X[:dev_id], X[dev_id:test_id], X[test_id:]
    ytrain, ydev, ytest = y[:dev_id], y[dev_id:test_id], y[test_id:]
    ids_train, ids_dev, ids_test = ids_tensor[:dev_id], ids_tensor[dev_id:test_id], ids_tensor[test_id:]
    tensors = [(Xtrain, ytrain, ids_train), (Xdev, ydev, ids_dev), (Xtest, ytest, ids_test)]
    ids_list = ids_list[test_id:]
    anns = anns[test_id:]

    for i, f in enumerate(tensor_files):
        with open(out_path + f, 'wb') as wf:
            pkl.dump(tensors[i], wf)
        print('splitted tensor out to {}, shape {}'.format(f, [t.shape for t in tensors[i]]))

    ids_file, anns_file = txt_files    
    with open(out_path + ids_file, 'w') as wf:
        for j in range(len(ids_list)):
            wf.write(str(ids_list[j]) + '\n')
    print('splitted test ids out to {}, len {}'.format(f, len(ids_list)))
    with open(out_path + anns_file, 'w') as wf:
        for j in range(len(anns)):
            wf.write(anns[j])
    print('splitted test anns out to {}, len {}'.format(f, len(anns)))


if __name__=="__main__":
    data_path = '/Users/sjx/Desktop/data/'
    out_path = '/Users/sjx/Desktop/out0506/'

    # load sample 
    sample_file = 'SAMPLE_MAY6.csv'
    sample = pd.read_csv(data_path + sample_file)
    pprint(sample.head())

    # # 1
    # code2index = codes_and_titles('codes_and_titles_256_0501.csv', 'code2index_256_0501.pkl')
    # # 2
    # # title_tensor_vocab('codes_and_titles_256_0501.csv', 'title2index_256_0501.pkl', 'title2tensor_256_0501.pkl')
    mlb, labels, anns = mlb('mlb_high_0505.pkl', 'anns_high_0505.txt')
    data, word2index = process_data_vocab('data_high_0505.pkl', 'word2index_high_0505.pkl')
    X, y, ids_tensor, ids_list = data_tensor(data, word2index, 'data_tensors_high_0505.pkl', 'ids_high_0505.txt')
    # 3
    tensor_files = ['train_tensor_high_0505.pkl', 'dev_tensor_high_0505.pkl', 'test_tensor_high_0505.pkl']
    txt_files = ['test_ids_high_0505.txt', 'test_anns_high_0505.txt']
    split(X, y, ids_tensor, ids_list, anns, 17000, 18000, tensor_files, txt_files)

    # misc
    label2icd = {0: '001-139',
                1: '140-239',
                2: '240-279',
                3: '280-289',
                4: '290-319',
                5: '320-389',
                6: '390-459',
                7: '460-519',
                8: '520-579',
                9: '580-629',
                10: '630-679',
                11: '680-709',
                12: '710-739',
                13: '740-759',
                14: '760-779',
                15: '780-799',
                16: '800-999',
                17: 'V01-V91',
                18: 'E000-E999'}

    # get diag_code 
    sample_high = pd.read_csv(data_path + 'May6.csv')
    sample_high.drop(inplace=True, columns=['padded_tokens', 'CLEANED TEXT', 'CLEAN_WORDS', 'CODED_TEXT', 'HIGH_LVL_DIAG'])
    print(sample_high.shape)

    diag_codes = []
    codes = sample_high['CODED_HIGH_LVL_DIAG'].values.tolist()
    for c in codes:
        c = sorted(list(set(map(int, c[1:-1].split(', ')))))
        c = ','.join([label2icd[c] for c in c])
        diag_codes.append(c)
    sample_high['DIAG_CODES'] = diag_codes
    pprint(sample_high.head())

    sample_high = sample_high[['ind', 'STEM_WORDS', 'DIAG_CODES']]
    sample_high.to_csv(data_path + 'SAMPLE_MAY6.csv')

    # # merge diag_code w/ sample
    # sample_256 = pd.read_csv(data_path + 'SAMPLE_256.csv')
    # sample.drop('Unnamed: 0')
    # sample = sample_256.merge(sample_high, on='HADM_ID')
    # print(sample.shape)

    # sample = sample[['HADM_ID', 'CATEGORY', 'TEXT', 'CLEAN_TEXT', 'CODED_TEXT', 'DIAG_CODES_y']]
    # sample.columns = ['HADM_ID', 'CATEGORY', 'TEXT', 'CLEAN_TEXT', 'CODED_TEXT', 'DIAG_CODES']
    # sample.to_csv(data_path + 'SAMPLE_HIGH.csv')

'''
OUTPUT:
(base) sjx@sjx-2:~/Desktop/dlpj$ python3 util.py
===== start generating codes_and_titles_20k_0425.csv, code2index_20k_0425.pkl =====
number of codes in SAMPLE_20K.csv:4902
number of codes in dict: 300
codes and titles out to: codes_and_titles_20k_0425.csv
len(code2index): 302
code2index out to: code2index_20k_0425.pkl
===== start generating title2index_20k_0425.pkl, title2tensor_20k_0425.pkl =====
title tensor shape: torch.Size([289, 10])
title2index: 291
title2idnex out to: title2index_20k_0425.pkl
title2tensor out to: title2tensor_20k_0425.pkl
===== start generating mlb_20k_0425.pkl, anns_20k_0425.txt =====
len(anns): 20000
anns out to: anns_20k_0425.txt
labels shape: (20000, 300)
number of classes: 300
mlb out to: mlb_20k_0425.pkl
===== start generating data_20k_0425.pkl, word2index_20k_0425.pkl =====
processed train data out to: data_20k_0425.pkl
vocab size: 10000
[('<pad>', 0), ('<unk>', 1), ('pt', 2), ('ml', 3), ('left', 4)]
word2index out to: word2index_20k_0425.pkl
===== start generating data_tensors_20k_0425.pkl, ids_20k_0425.txt =====
torch.Size([20000, 256]) torch.Size([20000, 300]) torch.Size([20000])
(X, y, ids) data tensors out to: data_tensors_20k_0425.pkl
ids out to: ids_20k_0425.txt
===== start generating splits =====
splitted tensor out to train_tensor_20k_0425.pkl, shape [torch.Size([18000, 256]), torch.Size([18000, 300]), torch.Size([18000])]
splitted tensor out to dev_tensor_20k_0425.pkl, shape [torch.Size([1000, 256]), torch.Size([1000, 300]), torch.Size([1000])]
splitted tensor out to test_tensor_20k_0425.pkl, shape [torch.Size([1000, 256]), torch.Size([1000, 300]), torch.Size([1000])]
splitted txt out to test_ids_20k_0425.txt, len 1000
splitted txt out to test_anns_20k_0425.txt, len 1000
'''


'''
after the above processing, main can start from dataloader

REQUIREMENTS:
vocabulary size ++ (10000)
sample size ++ (20k)
sample_after keep DIAG_CODES (not actually)
word2idx {'<pad>':0, '<unk>':1}

CHANGES:
evaluation.py: load_anns_dev
'''

