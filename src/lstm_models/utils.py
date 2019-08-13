import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .optim import AdamW
from ..data import TextDataset, LengthBucketingDataLoader
from ..metrics import JigsawEvaluator, accuracy
from ...config.base import IDENTITY_COLUMNS


def train(model: nn.Module, loss_fn: nn.Module, train_dataset: TextDataset, valid_dataset: TextDataset,
          device, batch_size=512, num_epochs=50, tolerance=10, lr=5e-3):
    start_time = time.time()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.000025)

    best_metrics = 0
    best_epoch = -1
    best_model_path = './best_model.bin'
    no_improvement_count = 0

    iteration_records = []
    for epoch in range(num_epochs):
        train_ps = []
        train_as = []
        train_ys = []
        train_yis = []

        model.train()
        for x_batch, _, a_batch, y_batch, y_identity_batch in tqdm(
                LengthBucketingDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
                total=len(train_dataset) // batch_size):
            x_batch = x_batch.type(torch.LongTensor).to(device)
            a_batch = a_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            p_batch = model(x_batch)
            loss = loss_fn(p_batch, y_batch, a_batch)
            loss.backward()
            optimizer.step()

            train_ps.append(p_batch.detach().cpu())
            train_as.append(a_batch.detach().cpu())
            train_ys.append(y_batch.detach().cpu())
            train_yis.append(y_identity_batch.detach().cpu())
        train_ps = torch.cat(train_ps, 0)
        train_as = torch.cat(train_as, 0)
        train_ys = torch.cat(train_ys, 0)
        train_yis = torch.cat(train_yis, 0)
        train_loss = loss_fn(train_ps, train_ys, train_as).item()
        train_evaluator = JigsawEvaluator(train_ys.numpy()[:, 0], train_yis.numpy())
        train_metrics, train_record = train_evaluator.get_final_metric(torch.sigmoid(train_ps[:, 0]).numpy())

        model.eval()
        with torch.no_grad():
            valid_ps = []
            valid_as = []
            valid_ys = []
            valid_yis = []
            for x_batch, _, a_batch, y_batch, y_identity_batch in tqdm(
                    LengthBucketingDataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False),
                    total=len(valid_dataset) // batch_size):
                x_batch = x_batch.type(torch.LongTensor).to(device)
                a_batch = a_batch.to(device)
                y_batch = y_batch.to(device)
                p_batch = model(x_batch)
                valid_ps.append(p_batch.detach().cpu())
                valid_as.append(a_batch.detach().cpu())
                valid_ys.append(y_batch.detach().cpu())
                valid_yis.append(y_identity_batch.detach().cpu())
        valid_ps = torch.cat(valid_ps, 0)
        valid_as = torch.cat(valid_as, 0)
        valid_ys = torch.cat(valid_ys, 0)
        valid_yis = torch.cat(valid_yis, 0)
        valid_loss = loss_fn(valid_ps, valid_ys, valid_as).item()
        valid_evaluator = JigsawEvaluator(valid_ys.numpy()[:, 0], valid_yis.numpy())
        valid_metrics, valid_record = valid_evaluator.get_final_metric(torch.sigmoid(valid_ps[:, 0]).numpy())

        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1}/{num_epochs}:\ttrn-loss={train_loss:.4f}\tval_loss={valid_loss:.4f}\t'
              f'trn-metrics={train_metrics}\tval-metrics={valid_metrics}\ttime={elapsed_time:.2f}s')

        iteration_records.append({
            'train': {
                'loss': train_loss,
                'accuracy': accuracy(train_ps[:, 0], train_ys[:, 0]),
                'metrics': {
                    **train_record
                }
            },
            'valid': {
                'loss': valid_loss,
                'accuracy': accuracy(valid_ps[:, 0], valid_ys[:, 0]),
                'metrics': {
                    **valid_record
                }
            }
        })

        if valid_metrics > best_metrics:
            best_metrics = valid_metrics
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count > tolerance:
            break

    print(f'Best: epoch={best_epoch + 1}\tmetrics={best_metrics}')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    os.remove(best_model_path)
    return model, iteration_records


def make_aggregated_score_table(records):
    scores = []
    for i, record in enumerate(records):
        for dataset_type in ['train', 'valid']:
            dataset_record = record[dataset_type]

            scores.append({
                'fold': i,
                'type': dataset_type,
                'loss': dataset_record['loss'],
                'accuracy': dataset_record['accuracy'],
                'mean_subgroup_auc': dataset_record['metrics']['mean_subgroup_auc'],
                'mean_bpsn_auc': dataset_record['metrics']['mean_bpsn_auc'],
                'mean_bnsp_auc': dataset_record['metrics']['mean_bnsp_auc'],
                'final_score': dataset_record['metrics']['final_score']
            })
    return pd.DataFrame(scores, columns=[
        'fold', 'type', 'final_score', 'loss', 'accuracy',
        'mean_subgroup_auc', 'mean_bpsn_auc', 'mean_bnsp_auc'])


def make_subgroup_score_table(records, score_type):
    scores = []    
    for i, record in enumerate(records):
        for dataset_type in ['train', 'valid']:
            dataset_record = record[dataset_type]
            scores.append({
                'fold': i,
                'type': dataset_type,
                **dataset_record['metrics'][f'{score_type}_auc'],
                'mean_score': dataset_record['metrics'][f'mean_{score_type}_auc'],
            })        
    return pd.DataFrame(scores, columns=['fold', 'type', 'mean_score'] + IDENTITY_COLUMNS)


def display_tables(records_dir: Path, verbose=True):

    def choose_best_record(fold_record):
        best_record = None
        best_score = 0
        for iter_record in fold_record:
            if iter_record['valid']['metrics']['final_score'] > best_score:
                best_score = iter_record['valid']['metrics']['final_score']
                best_record = iter_record
        return best_record

    def display_subgroup_tables(fold_records, score_type):
        subgroup_score_table = make_subgroup_score_table(
            best_records, score_type=score_type).query("type == 'valid'")
        if verbose:
            print(subgroup_score_table)
        print(subgroup_score_table.describe().loc[['mean', 'std'], subgroup_score_table.columns[2:]])

    fold_records = []
    for i in range(6):
        try: 
            fold_records.append(json.load(open(records_dir / f'records.{i}.json')))
        except FileNotFoundError:
            pass
    best_records = [choose_best_record(fold_record) for fold_record in fold_records]
        
    print("Score overview")
    agg_score_table = make_aggregated_score_table(best_records).query("type == 'valid'")
    if verbose:
        print(agg_score_table)
    print(agg_score_table.describe().loc[['mean', 'std'], agg_score_table.columns[2:]])
    
    print("Subgroup AUCs")
    display_subgroup_tables(best_records, 'subgroup')
    
    print("BNSP AUCs")
    display_subgroup_tables(best_records, 'bnsp')

    print("BPSN AUCs")
    display_subgroup_tables(best_records, 'bpsn')


class EMA(object):

    def __init__(self, model, mu, level='batch', n=1):
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.level = level
        self.n = n
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level is 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n

    def on_epoch_end(self, model):
        if self.level is 'epoch':
            self._update(model)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def eval_model(model, data_loader, device):
    model.eval()
    preds_fold = np.zeros(len(data_loader.dataset))

    with torch.no_grad():
        for index, x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch).detach()
            preds_fold[list(index)] = sigmoid(y_pred.cpu().numpy())[:, 0]

    return preds_fold
