from pytorch_pretrained_bert import GPT2Model
import torch
from torch import nn


class GPT2ClassificationHeadModel(GPT2Model):

    def __init__(self, config, clf_dropout=0.4, n_class=8):
        super(GPT2ClassificationHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(config.n_embd * 3, n_class)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)
        h_conc = torch.cat((avg_pool, max_pool, hidden_states[:, -1, :]), 1)
        logits = self.linear(self.dropout(h_conc))
        return logits
