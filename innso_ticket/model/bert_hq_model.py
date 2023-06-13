import numpy as np
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch import nn
class bert_model(BertPreTrainedModel):
    def __init__(self, config, levels_num):
        super(bert_model, self).__init__(config)
        self.labels = levels_num
        self.levels_num = len(np.unique(np.array(levels_num)))
        self.bert = BertModel(config)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.level = nn.Linear(config.hidden_size, len(levels_num))
        self.loss = nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, mask_ids, seg_ids, labels=None):
        _, embed_output = self.bert(input_ids, mask_ids, seg_ids)
        embed_output = self.drop(embed_output)
        logits = self.level(embed_output)
        if labels is None:
            return logits
        labels = labels.view(-1)
        loss = self.loss(logits.view(-1, self.levels_num), labels)
        return loss, logits

    # def forward(self, input_ids, mask_ids, seg_ids, labels=None):
    #     _, embed_output = self.bert(input_ids, mask_ids, seg_ids)
    #     embed_output = self.drop(embed_output)
    #     logits = self.level(embed_output)
    #     return logits