import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, f1_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
MAX_SEQ_LEN = 512
num_labels = 19
lr = 5e-5
max_grad_norm = 1.0
num_training_steps = 500  # TODO
num_warmup_steps = max(1, num_training_steps // 10)
train_data_file = "data/clean.csv"
model_ckpt = 'model0508.ckpt'

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir='./model_ckpt/')
clinic_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir='./model_ckpt/')


class ClinicBertMultiLabelClassifier(pl.LightningModule):
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

    @staticmethod
    def _convert_label_id_to_one_hot(code_list):
        labels = np.zeros([len(code_list), num_labels])
        for idx, codes in enumerate(code_list):
            codes = eval(codes)
            if not isinstance(codes, list):
                print("NOT a list: ", idx, codes)
            for code in codes:
                labels[idx][code] = 1
        return torch.tensor(labels)

    @staticmethod
    def _get_pos_weight(labels: torch.Tensor) -> torch.Tensor:
        total_num = labels.size(0)
        pos_cnt = labels.sum(dim=0).cpu().detach().numpy()
        neg_cnt = total_num - pos_cnt
        tmp = neg_cnt / pos_cnt
        return torch.tensor(tmp).to(device)

    def prepare_data(self):
        df = pd.read_csv(train_data_file)
        input_sequence_list = df['CLEAN_WORDS']
        input_data = tokenizer.batch_encode_plus([" ".join(eval(e)) for e in input_sequence_list],
                                                 max_length=MAX_SEQ_LEN,
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        input_ids = input_data['input_ids']  # IntTensor [batch_size, MAX_SEQ_LEN]
        print('input_ids: ', type(input_ids), input_ids.shape)
        code_list = df['CODED_HIGH_LVL_DIAG']
        labels = self._convert_label_id_to_one_hot(code_list)

        # re-define loss function with weights
        self.pos_weight = self._get_pos_weight(labels)
        print(self.pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        print('labels: ', type(labels), labels.shape)
        self.train_dataset = TensorDataset(input_ids[:17000], labels[:17000])
        self.val_dataset = TensorDataset(input_ids[17000:], labels[17000:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}
        # return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        batch_loss, probs = self(input_ids, labels=labels)

        probs = probs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype(int)
        labels_pred = np.round(probs).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, labels_pred, average='micro')
        f1_macro = f1_score(labels, labels_pred, average='macro')
        accuracy = accuracy_score(labels, labels_pred)

        log = {'train_loss': batch_loss}
        return {'loss': batch_loss, 'log': log,
                'precision': torch.tensor(precision),
                'recall': torch.tensor(recall),
                'f1': torch.tensor(f1),
                'f1-macro': torch.tensor(f1_macro),
                'accuracy': torch.tensor(accuracy),
                }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['f1'] for x in outputs]).mean()
        avg_f1_macro = torch.stack([x['f1-macro'] for x in outputs]).mean()
        avg_precision = torch.stack([x['precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['recall'] for x in outputs]).mean()
        print('Train F1', avg_f1.item())
        print('Train precision', avg_precision.item())
        print('Train recall', avg_recall.item())
        print('Train acc: ', avg_acc.item())

        self.train_loss_list.append(avg_loss)
        self.train_f1_micro_list.append(avg_f1)
        self.train_f1_macro_list.append(avg_f1_macro)
        self.train_precision_list.append(avg_precision)
        self.train_recall_list.append(avg_recall)

        return {'loss': avg_loss}

    def _precision_top_n(self, probs_pred: np.ndarray, labels_true: np.ndarray, n: int):
        # TODO
        # [batch size, n_classes]
        prob_ids = probs_pred.argsort(axis=1)[:, ::-1][:, :n]  # reverse and trunc to only top n
        top_preds = np.zeros_like(labels_true)
        for i, prob_id in enumerate(prob_ids):
            top_preds[i][prob_id] = 1
        return precision_score(labels_true, top_preds, average='micro')

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        loss, logits = self(input_ids, labels=labels)
        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype(int)
        # p8 = self._precision_top_n(logits, labels, 8)
        # p15 = self._precision_top_n(logits, labels, 15)
        labels_pred = np.round(logits).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, labels_pred, average='micro')
        f1_macro = f1_score(labels, labels_pred, average='macro')
        accuracy = accuracy_score(labels, labels_pred)
        return {'precision': torch.tensor(precision),
                'recall': torch.tensor(recall),
                'f1': torch.tensor(f1),
                'f1-macro': torch.tensor(f1_macro),
                'accuracy': torch.tensor(accuracy),
                'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['f1'] for x in outputs]).mean()
        avg_f1_macro = torch.stack([x['f1-macro'] for x in outputs]).mean()
        avg_precision = torch.stack([x['precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['recall'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_loss_list.append(avg_loss)
        self.val_f1_micro_list.append(avg_f1)
        self.val_f1_macro_list.append(avg_f1_macro)
        self.val_precision_list.append(avg_precision)
        self.val_recall_list.append(avg_recall)
        print('Val Loss', avg_loss.item())
        print('Val F1', avg_f1.item())
        print('Val precision', avg_precision.item())
        print('Val recall', avg_recall.item())
        print('Val acc: ', avg_acc.item())
        f1 = avg_f1.item()
        if f1 > self.best_f1:
            self.best_f1 = f1
            model_dict = {
                'f1': f1,
                'model': self.state_dict(),
                'train_loss': self.train_loss_list,
                'val_loss': self.val_loss_list,
                'train_f1': self.train_f1_micro_list,
                'val_f1': self.val_f1_micro_list,
                'train_f1_macro': self.train_f1_macro_list,
                'val_f1_macro': self.val_f1_macro_list,
                'train_precision': self.train_precision_list,
                'train_recall': self.train_recall_list,
                'val_precision': self.val_precision_list,
                'val_recall': self.val_recall_list,
            }
            torch.save(model_dict, model_ckpt)
            print("Save model at f1[%f] in %s" % (f1, model_ckpt))
        return {'val_loss': avg_acc,
                'val_avg_f1': avg_f1}


if __name__ == '__main__':
    net = ClinicBertMultiLabelClassifier()
    trainer = pl.Trainer(max_epochs=num_training_steps, gpus=1)

    if os.path.exists(model_ckpt):
        print("Load model from %s" % model_ckpt)
        checkpoint = torch.load(model_ckpt)
        net.load_state_dict(checkpoint['model'])
        net.best_f1 = checkpoint['f1']
        print("Previous best val f1: %f" % net.best_f1)
        net.val_loss_list = list(checkpoint['val_loss'])
        net.train_loss_list = list(checkpoint['train_loss'])
    trainer.fit(net)

    model_dict = {
        'model': net.state_dict(),
        'train_loss': net.train_loss_list,
        'val_loss': net.val_loss_list,
        'train_f1': net.train_f1_micro_list,
        'val_f1': net.val_f1_micro_list,
        'train_f1_macro': net.train_f1_macro_list,
        'val_f1_macro': net.val_f1_macro_list,
        'train_precision': net.train_precision_list,
        'train_recall': net.train_recall_list,
        'val_precision': net.val_precision_list,
        'val_recall': net.val_recall_list
    }
    torch.save(model_dict, 'model0508-whole.ckpt')
    print("Save whole process in model0508-whole.ckpt")
