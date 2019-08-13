import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..data import TextDataset, LengthBucketingDataLoader


def get_optimizer_params(model, lr, lr_weight_decay_coef, num_layers):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if lr_weight_decay_coef < 1.0:
        optimizer_grouped_parameters = [
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' not in n
                and 'bert.encoder' not in n
                and not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' not in n
                and 'bert.encoder' not in n
                and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' in n
                and not any(nd in n for nd in no_decay)],
             'lr': lr * lr_weight_decay_coef ** (num_layers + 1), 'weight_decay': 0.01},
            {'params': [
                p for n, p in param_optimizer
                if 'bert.embeddings' in n
                and any(nd in n for nd in no_decay)],
             'lr': lr * lr_weight_decay_coef ** (num_layers + 1), 'weight_decay': 0.0}
        ]
        for i in range(num_layers):
            optimizer_grouped_parameters.append(
                {'params': [
                    p for n, p in param_optimizer
                    if 'bert.encoder.layer.{}.'.format(i) in n
                    and any(nd in n for nd in no_decay)],
                 'lr': lr * lr_weight_decay_coef ** (num_layers - i), 'weight_decay': 0.0})
            optimizer_grouped_parameters.append(
                {'params': [
                    p for n, p in param_optimizer
                    if 'bert.encoder.layer.{}.'.format(i) in n
                    and any(nd in n for nd in no_decay)],
                 'lr': lr * lr_weight_decay_coef ** (num_layers - i), 'weight_decay': 0.0})
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    return optimizer_grouped_parameters


def predict(model: nn.Module, dataset: TextDataset, device, batch_size=32) -> np.ndarray:
    model.eval()
    test_ps = []
    test_is = []
    with torch.no_grad():
        for batch in tqdm(LengthBucketingDataLoader(dataset=dataset, batch_size=batch_size,
                                                    shuffle=False, drop_last=False),
                          total=len(dataset) // batch_size):
            x_batch = batch[0]
            i_batch = batch[1]
            p_batch = model(x_batch.type(torch.LongTensor).to(device))
            test_ps.append(p_batch.detach().cpu())
            test_is.append(i_batch.detach().cpu())
    test_ps = torch.sigmoid(torch.cat(test_ps, 0)[:, 0]).numpy().ravel()
    test_is = torch.cat(test_is, 0).numpy().ravel()
    return np.array(list(map(lambda pi: pi[0], sorted(list(zip(test_ps, test_is)), key=lambda pi: pi[1]))))
