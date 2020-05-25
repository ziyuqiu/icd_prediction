import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, f1_score
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
MAX_SEQ_LEN = 256
num_labels = 300
lr = 5e-5
max_grad_norm = 1.0
num_training_steps = 500  # TODO
num_warmup_steps = max(1, num_training_steps // 10)
train_data_file = "data/data.csv"
model_ckpt = 'model-val.ckpt'

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir='./model_ckpt/')
clinic_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir='./model_ckpt/')


class ClinicBertMultiLabelClassifier(nn.Module):
    def __init__(self, grad_clip=True):
        super(ClinicBertMultiLabelClassifier, self).__init__()
        self.grad_clip = grad_clip
        self.num_labels = num_labels

        # evaluation metrics
        self.best_f1 = 0
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_f1_micro_list = []
        self.train_f1_macro_list = []
        self.val_f1_micro_list = []
        self.val_f1_macro_list = []
        self.train_precision_list = []
        self.train_recall_list = []
        self.val_precision_list = []
        self.val_recall_list = []

        # loss function
        self.pos_weight = torch.ones([num_labels]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # network modules
        self.bert = clinic_bert
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.activate = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, aggregated_output = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(aggregated_output)

        # to avoid gradients vanishing and sigmoid nan
        if self.grad_clip:
            logits = logits.clamp(min=-14.0, max=14.0)

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.reshape([-1, self.num_labels])[active_loss]
                active_labels = labels.reshape([-1, self.num_labels])[active_loss]
                loss = self.criterion(active_logits, active_labels)
            else:
                loss = self.criterion(logits.reshape([-1, self.num_labels]),
                                      labels.reshape([-1, self.num_labels]))
            return loss, self.activate(logits)
        else:
            return self.activate(logits)


def convert_label_id_to_one_hot(code_list):
    labels = np.zeros([len(code_list), num_labels])
    for idx, codes in enumerate(code_list):
        codes = eval(codes)
        if not isinstance(codes, list):
            print("NOT a list: ", idx, codes)
        for code in codes:
            labels[idx][code] = 1
    return torch.tensor(labels)


def prepare_data():
    df = pd.read_csv(train_data_file)
    input_sequence_list = df['CLEAN_WORDS']
    input_data = tokenizer.batch_encode_plus([" ".join(eval(e)) for e in input_sequence_list],
                                             max_length=MAX_SEQ_LEN,
                                             pad_to_max_length=True,
                                             return_tensors='pt')
    input_ids = input_data['input_ids']  # IntTensor [batch_size, MAX_SEQ_LEN]
    print('input_ids: ', type(input_ids), input_ids.shape)
    code_list = df['CODED_DIAG']
    labels = convert_label_id_to_one_hot(code_list)

    print('labels: ', type(labels), labels.shape)
    train_dataset = TensorDataset(input_ids[:17000], labels[:17000])
    val_dataset = TensorDataset(input_ids[17000:], labels[17000:])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_dataloader, val_dataloader


def precision_top_n(probs_pred: np.ndarray, labels_true: np.ndarray, n: int):
    # [batch size, n_classes]
    prob_ids = probs_pred.argsort(axis=1)[:, ::-1][:, :n]  # reverse and trunc to only top n
    top_preds = np.zeros_like(labels_true)
    for i, prob_id in enumerate(prob_ids):
        top_preds[i][prob_id] = 1
    return precision_score(labels_true, top_preds, average='micro')


if __name__ == '__main__':
    net = ClinicBertMultiLabelClassifier()
    if os.path.exists(model_ckpt):
        print("Load model from %s" % model_ckpt)
        checkpoint = torch.load(model_ckpt, map_location=device)
        net.load_state_dict(checkpoint['model'])
        net.best_f1 = checkpoint['f1']
        print("Previous best val f1: %f" % net.best_f1)

    _, dataloader = prepare_data()
    print("Data prepared.")

    net.eval()
    p8_list = []
    p15_list = []
    f1_list = []
    f1_macro_list = []
    precision_list = []
    recall_list = []
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader):
            _, logits = net(input_ids, labels=labels)
            probs = logits.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy().astype(int)
            labels_pred = np.round(probs).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, labels_pred, average='micro')
            f1_macro = f1_score(labels, labels_pred, average='macro')
            p8 = precision_top_n(probs, labels, 8)
            p15 = precision_top_n(probs, labels, 15)
            f1_list.append(f1)
            f1_macro_list.append(f1_macro)
            precision_list.append(precision)
            recall_list.append(recall)
            p8_list.append(p8)
            p15_list.append(p15)
        p8_list = np.array(p8_list)
        p15_list = np.array(p15_list)
        f1_list = np.array(f1_list)
        f1_macro_list = np.array(f1_macro_list)
        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        print("F1: ", f1)
        print("f1-macro: ", f1_macro)
        print("precision: ", precision)
        print("recall: ", recall)
        print("pre@8: ", p8)
        print("pre@15: ", p15)
