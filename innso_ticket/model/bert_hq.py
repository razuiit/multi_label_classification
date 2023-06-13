import os

from typing import List, Tuple
import collections
import numpy as np
import tensorflow as tf

import torch
from pytorch_pretrained_bert import (BertModel, BertTokenizer)
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from innso_ticket.model.bert_hq_model import bert_model
from innso_ticket import config
from innso_ticket.model.base import BaseModel
from innso_ticket.unit.data import Data
from innso_ticket.util import logutil


logger = logutil.logger_run

tf.random.set_seed(config.SEED)

MAX_LENGTH = 128
BATCH_SIZE = 16
VALIDATION_SIZE = 0.15
epochs = 1


class BERT_HQ(BaseModel):
    def __init__(self,
                 scope_name: str,
                 full_types: List[str] = None) -> None:
        super(BERT_HQ, self).__init__(scope_name, full_types)

        self.tokenizer = BertTokenizer(config.PRETRAINED_MODEL_DIR+"/vocab.txt")

        if self.full_types is not None:
            self.num_types = len(self.full_types)
            self.model = bert_model.from_pretrained(os.path.join(config.PRETRAINED_MODEL_DIR, config.BERT_MODEL_HQ),
                                                    self.full_types)
            self.optimizer = None

    def __calc_full_type(self,
                         y_pred: np.ndarray) -> List[str]:
        return [self.full_types[i] for i in np.argmax(y_pred, axis=1)]

    def tokenize(self,
                 text, tokenizer):
        text = str(text).strip().lower()
        tok_ids = tokenizer.tokenize(text)
        if len(tok_ids) > MAX_LENGTH - 2:
            tok_ids = tok_ids[:MAX_LENGTH - 2]
        tok_ids.insert(0, "[CLS]")
        tok_ids.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tok_ids)
        mask_ids = [1] * len(input_ids)
        seg_ids = [0] * len(input_ids)
        padding = [0] * (MAX_LENGTH - len(input_ids))
        input_ids += padding
        mask_ids += padding
        seg_ids += padding
        return input_ids, mask_ids, seg_ids

    def get_sample(self,
                   data):
        data_list = data.X_text.tolist()
        label_list = data.y_index
        _input, _mask, _seg, _label = [], [], [], []
        for i in range(len(data_list)):
            a, b, c = self.tokenize(data_list[i], self.tokenizer)
            _input.append(a)
            _mask.append(b)
            _seg.append(c)
            _label.append(label_list[i])
        _input = np.array(_input)
        _mask = np.array(_mask)
        _seg = np.array(_seg)
        return _input, _mask, _seg, _label

    def get_sample_pred(self,
                   data):
        data_list = data.X_text.tolist()
        _input, _mask, _seg = [], [], []
        for i in range(len(data_list)):
            a, b, c = self.tokenize_pred(data_list[i], self.tokenizer,  MAX_LENGTH)
            _input.append(a)
            _mask.append(b)
            _seg.append(c)
        _input = np.array(_input)
        _mask = np.array(_mask)
        _seg = np.array(_seg)
        return _input, _mask, _seg

    def tokenize_pred(self, text, tokenizer, max_seq=100):
        text = text.strip().lower()
        tok_ids = tokenizer.tokenize(text)
        if len(tok_ids) > max_seq - 2:
            tok_ids = tok_ids[:max_seq - 2]
        tok_ids.insert(0, "[CLS]")
        tok_ids.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tok_ids)
        mask_ids = [1] * len(input_ids)
        seg_ids = [0] * len(input_ids)
        padding = [0] * (max_seq - len(input_ids))
        input_ids += padding
        mask_ids += padding
        seg_ids += padding
        return input_ids, mask_ids, seg_ids

    def get_labels(self, train, dev):
        label_to_num = collections.OrderedDict()
        tmp = train.y_index + dev.y_index
        x = np.array(tmp)
        a1 = (np.unique(x))
        n = 0
        for kk in a1:
            label_to_num[kk] = n
            n += 1
        levels_num = len(a1)
        return label_to_num, levels_num

    def pred_acc(self, logits, labels):
        acc_1 = acc_2 = acc_3 = acc_4 = 0.0
        pred_res = torch.argmax(logits, 1)
        cred = torch.max(torch.nn.functional.softmax(logits), 1)[0]
        cred = cred > 0.5
        count = 0
        pred_list = []
        for i in range(labels.size()[0]):
            y = self.full_types[int(labels[i].cpu().numpy())].split('^')
            y_pred = self.full_types[int(pred_res[i].cpu().numpy())].split('^')
            pred_list.append("^".join(y_pred))
            if y[0] == y_pred[0]:
                acc_1 += 1
                if cred[i]:
                    count += 1
                if y[1] == y_pred[1]:
                    acc_2 += 1
                    if y[2] == y_pred[2]:
                        acc_3 += 1
                        if len(y) > 3 and len(y_pred) > 3 and y[3] == y_pred[3]:
                            acc_4 += 1
        acc_res = [acc_1, acc_2, acc_3, acc_4, count, torch.sum(cred).cpu().numpy()]
        return acc_res, pred_list

    def train(self,
              data_train: Data,
              data_test: Data) -> Tuple[float, List[float]]:
        input_train, mask_train, seg_train, label_train = self.get_sample(data_train)
        input_train = torch.tensor(input_train, dtype=torch.long)
        mask_train = torch.tensor(mask_train, dtype=torch.long)
        seg_train = torch.tensor(seg_train, dtype=torch.long)
        label_train = torch.tensor(label_train, dtype=torch.long)
        train_data = TensorDataset(input_train, mask_train, seg_train, label_train)
        sample = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=sample, batch_size=BATCH_SIZE)

        input_test, mask_test, seg_test, label_test = self.get_sample(data_test)
        input_test = torch.tensor(input_test, dtype=torch.long)
        mask_test = torch.tensor(mask_test, dtype=torch.long)
        seg_test = torch.tensor(seg_test, dtype=torch.long)
        label_test = torch.tensor(label_test, dtype=torch.long)
        test_data = TensorDataset(input_test, mask_test, seg_test, label_test)
        self.test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

        max_acc = 0
        for _ in range(3):
            device = torch.device('cpu')
            self.model.to(device)
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}]
            self.optimizer = BertAdam(optimizer_parameters, lr=2e-5)
            res = []
            for epoch in range(epochs):
                self.model.train()
                for step, batch in enumerate(self.train_dataloader):
                    # batch = tuple(t.cuda() for t in batch)
                    batch = tuple(t.to(device) for t in batch)
                    a, b, c, labels = batch
                    self.optimizer.zero_grad()
                    loss, logits = self.model(a, b, c, labels)
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()
                pred_eval = np.zeros(6)
                pred_labels = []
                with torch.no_grad():
                    test_loss = 0.
                    for step, batch in enumerate(self.test_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        a, b, c, labels = batch
                        loss, logits = self.model(a, b, c, labels)
                        res, pred = self.pred_acc(logits, labels)
                        pred_eval = pred_eval + np.array(res)
                        test_loss += loss
                        pred_labels.extend(pred)

                    res.append(pred_eval)
                    if pred_eval[2] > max_acc:
                        max_acc = pred_eval[2]
        y_true = data_test.y_type.tolist()
        return self._calc_accuracies(y_true, pred_labels)

    def predict_topk(self, logits, topk=2, credible=0.2):
        pred = torch.argmax(logits, 1).numpy()
        tmp = torch.sort(torch.nn.functional.softmax(logits, 1), 1, descending=True)
        prob = tmp[0].cpu().numpy()[:, :topk]
        cred = [True] * len(prob)
        for i in range(len(prob)):
            diff = prob[i][0] - prob[i][1]
            if diff < credible:
                cred[i] = False
        return pred, cred

    def predict(self,
                data_test: Data) -> List[str]:
        input_test, mask_test, seg_test = self.get_sample_pred(data_test)
        input_test = torch.tensor(input_test, dtype=torch.long)
        mask_test = torch.tensor(mask_test, dtype=torch.long)
        seg_test = torch.tensor(seg_test, dtype=torch.long)
        test_data = TensorDataset(input_test, mask_test, seg_test)
        self.test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


        device = torch.device('cpu')
        self.model.to(device)
        self.model.eval()
        pred_labels = []
        with torch.no_grad():
            for step, batch in enumerate(self.test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                a, b, c = batch
                logits = self.model(a, b, c, labels=None)
                pred, cred = self.predict_topk(logits, 2, 0.25)# credible record are with confidence more than 25%
                for i in range(len(pred)):
                    label = self.full_types[pred[i]]

                    pred_labels.append(label)

        return pred_labels

    def save_model(self):
        state = {
            'epoch': epochs,
            'model': self.model,
            'optimizer': self.optimizer,
            'classes': self.full_types}
        torch.save(state, os.path.join(config.MODEL_DIR, config.BERT_MODEL_HQ + '_' + self.scope_name))
        logger.info(
            f"Model has been saved at {os.path.join(config.MODEL_DIR, config.BERT_MODEL_HQ + '_' + self.scope_name)}")

    def load_model(self) -> None:
        checkpoint = torch.load(os.path.join(config.MODEL_DIR, config.BERT_MODEL_HQ + '_' + self.scope_name),
                                map_location='cpu')
        self.model = checkpoint.get("model")
        logger.info(
            f"Model is loaded from {os.path.join(config.MODEL_DIR, config.BERT_MODEL_HQ + '_' + self.scope_name)}")

