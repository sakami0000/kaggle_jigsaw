from argparse import ArgumentParser
import copy
from functools import partial
from itertools import chain
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.base import (
    TOXICITY_COLUMN, IDENTITY_COLUMNS, AUX_TOXICITY_COLUMNS,
    EMBEDDING_FASTTEXT, TRAIN_DATA, TEST_DATA, SAMPLE_SUBMISSION)

from src.data import TokenDataset, collate_fn, BucketSampler
from src.metrics import JigsawEvaluator
from src.utils import timer, seed_torch, send_line_notification, load_config
from src.weights import training_weights_s

from src.lstm_models.load_data import load_embedding
from src.lstm_models.optim import ParamScheduler
from src.lstm_models.models import LstmGruModel, LstmCapsuleAttenModel, LstmConvModel
from src.lstm_models.preprocess import preprocess
from src.lstm_models.tokenize import build_vocab, tokenize
from src.lstm_models.utils import EMA, sigmoid, eval_model


def main():
    parser = ArgumentParser()
    parser.add_argument('--lstm_model', type=str, required=True,
                        choices=['lstm_gru', 'lstm_capsule_atten', 'lstm_conv'])
    parser.add_argument('--valid', action='store_true')
    args = parser.parse_args()

    config = load_config('./config/lstm_s.json')
    config.setdefault('max_len', 220)
    config.setdefault('max_features', 100000)
    config.setdefault('batch_size', 512)
    config.setdefault('train_epochs', 6)
    config.setdefault('n_splits', 5)
    config.setdefault('start_lr', 1e-4)
    config.setdefault('max_lr', 5e-3)
    config.setdefault('last_lr', 1e-3)
    config.setdefault('warmup', 0.2)
    config.setdefault('pseudo_label', True)
    config.setdefault('mu', 0.9)
    config.setdefault('updates_per_epoch', 10)
    config.setdefault('lstm_gru', {})
    config.setdefault('lstm_capsule_atten', {})
    config.setdefault('lstm_conv', {})
    config.setdefault('device', 'cuda')
    config.setdefault('seed', 1234)

    device = torch.device(config.device)

    OUT_DIR = Path(f'../output/{args.lstm_model}/')
    MODEL_STATE = OUT_DIR / 'pytorch_model.bin'
    submission_file_name = 'valid_submission.csv' if args.valid else 'submission.csv'
    SUBMISSION_PATH = OUT_DIR / submission_file_name
    OUT_DIR.mkdir(exist_ok=True)

    warnings.filterwarnings('ignore')
    seed_torch(config.seed)

    if args.lstm_model == 'lstm_gru':
        neural_net = LstmGruModel
    elif args.lstm_model == 'lstm_capsule_atten':
        neural_net = LstmCapsuleAttenModel
        config.lstm_capsule_atten['max_len'] = config.max_len
    else:
        neural_net = LstmConvModel

    with timer('preprocess'):
        train = pd.read_csv(TRAIN_DATA, index_col='id')
        if args.valid:
            train = train.sample(frac=1, random_state=1029).reset_index(drop=True)
            test = train.tail(200000)
            train = train.head(len(train) - 200000)
        else:
            test = pd.read_csv(TEST_DATA)

        train['comment_text'] = train['comment_text'].apply(preprocess)
        test['comment_text'] = test['comment_text'].apply(preprocess)
        
        # replace blank with nan
        train['comment_text'].replace('', np.nan, inplace=True)
        test['comment_text'].replace('', np.nan, inplace=True)

        # nan prediction
        nan_pred = train['target'][train['comment_text'].isna()].mean()

        # fill up the missing values
        train_x = train['comment_text'].fillna('_##_').values
        test_x = test['comment_text'].fillna('_##_').values
        
        # get the target values
        weights = training_weights_s(train, TOXICITY_COLUMN, IDENTITY_COLUMNS)
        train_y = np.vstack([train[TOXICITY_COLUMN].values, weights]).T
        train_y_identity = train[IDENTITY_COLUMNS].values

        train_nan_mask = train_x == '_##_'
        test_nan_mask = test_x == '_##_'
        y_binary = (train_y[:, 0] >= 0.5).astype(int)
        y_identity_binary = (train_y_identity >= 0.5).astype(int)
        
        vocab = build_vocab(chain(train_x, test_x), config.max_features)
        embedding_matrix = load_embedding(EMBEDDING_FASTTEXT, vocab['token2id'])

        joblib.dump(vocab, OUT_DIR / 'vocab.pkl')
        np.save('embedding_matrix', embedding_matrix)

        train_x = np.array(tokenize(train_x, vocab, config.max_len))
        test_x = np.array(tokenize(test_x, vocab, config.max_len))

        models = {}
        train_preds = np.zeros((len(train_x)))
        test_preds = np.zeros((len(test_x)))
        ema_train_preds = np.zeros((len(train_x)))
        ema_test_preds = np.zeros((len(test_x)))

    if config.pseudo_label:
        with timer('pseudo label'):
            train_dataset = TokenDataset(train_x, targets=train_y, maxlen=config.max_len)
            test_dataset = TokenDataset(test_x, maxlen=config.max_len)

            train_sampler = BucketSampler(train_dataset, train_dataset.get_keys(),
                                        bucket_size=config.batch_size * 20, batch_size=config.batch_size)
            test_sampler = BucketSampler(test_dataset, test_dataset.get_keys(),
                                        batch_size=config.batch_size, shuffle_data=False)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                    sampler=train_sampler, num_workers=0, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler,
                                    shuffle=False, num_workers=0, collate_fn=collate_fn)

            model = neural_net(embedding_matrix, **config[args.lstm_model]).to(device)

            ema_model = copy.deepcopy(model)
            ema_model.eval()

            ema_n = int(len(train_loader.dataset) / (config.updates_per_epoch * config.batch_size))
            ema = EMA(model, config.mu, n=ema_n)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            scheduler = ParamScheduler(optimizer, config.train_epochs * len(train_loader),
                                       start_lr=config.start_lr, max_lr=config.max_lr,
                                       last_lr=config.last_lr, warmup=config.warmup)

            all_test_preds = []
            
            for epoch in range(config.train_epochs):
                start_time = time.time()
                model.train()

                for _, x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    scheduler.batch_step()
                    y_pred = model(x_batch)

                    loss = nn.BCEWithLogitsLoss(weight=y_batch[:, 1])(y_pred[:, 0], y_batch[:, 0])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ema.on_batch_end(model)

                elapsed_time = time.time() - start_time
                print('Epoch {}/{} \t time={:.2f}s'.format(
                    epoch + 1, config.train_epochs, elapsed_time))

                all_test_preds.append(eval_model(model, test_loader))
                ema.on_epoch_end(model)

            ema.set_weights(ema_model)
            ema_model.lstm.flatten_parameters()
            ema_model.gru.flatten_parameters()

            checkpoint_weights = np.array([2 ** epoch for epoch in range(config.train_epochs)])
            checkpoint_weights = checkpoint_weights / checkpoint_weights.sum()

            ema_test_y = eval_model(ema_model, test_loader)
            test_y = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
            test_y = np.mean([test_y, ema_test_y], axis=0)
            test_y[test_nan_mask] = nan_pred
            weight = np.ones((len(test_y)))
            test_y = np.vstack((test_y, weight)).T

            models['model'] = model.state_dict()
            models['ema_model'] = ema_model.state_dict()

    with timer('train'):
        splits = list(
            StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed).split(train_x, y_binary))
        
        if config.pseudo_label:
            splits_test = list(KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed).split(test_x))
            splits = zip(splits, splits_test)

        for fold, split in enumerate(splits):
            print(f'Fold {fold + 1}')

            if config.pseudo_label:
                (train_idx, valid_idx), (train_idx_test, _) = split
                x_train_fold = np.concatenate((train_x[train_idx], test_x[train_idx_test]), axis=0)
                y_train_fold = np.concatenate((train_y[train_idx], test_y[train_idx_test]), axis=0)
            else:
                train_idx, valid_idx = split
                x_train_fold = train_x[train_idx]
                y_train_fold = train_y[train_idx]

            x_valid_fold = train_x[valid_idx]
            y_valid_fold = train_y[valid_idx]

            valid_nan_mask = train_nan_mask[valid_idx]

            y_valid_fold_binary = y_binary[valid_idx]
            y_valid_fold_identity_binary = y_identity_binary[valid_idx]
            evaluator = JigsawEvaluator(y_valid_fold_binary, y_valid_fold_identity_binary)

            train_dataset = TokenDataset(x_train_fold, targets=y_train_fold, maxlen=config.max_len)
            valid_dataset = TokenDataset(x_valid_fold, targets=y_valid_fold, maxlen=config.max_len)

            train_sampler = BucketSampler(train_dataset, train_dataset.get_keys(),
                                          bucket_size=config.batch_size * 20, batch_size=config.batch_size)
            valid_sampler = BucketSampler(valid_dataset, valid_dataset.get_keys(),
                                          batch_size=config.batch_size, shuffle_data=False)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                      sampler=train_sampler, num_workers=0, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                                      sampler=valid_sampler, collate_fn=collate_fn)

            model = neural_net(embedding_matrix, **config[args.lstm_model]).to(device)

            ema_model = copy.deepcopy(model)
            ema_model.eval()

            ema_n = int(len(train_loader.dataset) / (config.updates_per_epoch * config.batch_size))
            ema = EMA(model, config.mu, n=ema_n)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            scheduler = ParamScheduler(optimizer, config.train_epochs * len(train_loader),
                                       start_lr=config.start_lr, max_lr=config.max_lr,
                                       last_lr=config.last_lr, warmup=config.warmup)

            all_valid_preds = []
            all_test_preds = []

            for epoch in range(config.train_epochs):
                start_time = time.time()
                model.train()

                for _, x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    scheduler.batch_step()
                    y_pred = model(x_batch)

                    loss = nn.BCEWithLogitsLoss(weight=y_batch[:, 1])(y_pred[:, 0], y_batch[:, 0])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ema.on_batch_end(model)

                valid_preds = eval_model(model, valid_loader)
                valid_preds[valid_nan_mask] = nan_pred
                all_valid_preds.append(valid_preds)

                auc_score, _ = evaluator.get_final_metric(valid_preds)
                elapsed_time = time.time() - start_time
                print('Epoch {}/{} \t auc={:.5f} \t time={:.2f}s'.format(
                    epoch + 1, config.train_epochs, auc_score, elapsed_time))

                all_test_preds.append(eval_model(model, test_loader))

                models[f'model_{fold}{epoch}'] = model.state_dict()
                ema.on_epoch_end(model)

            ema.set_weights(ema_model)
            ema_model.lstm.flatten_parameters()
            ema_model.gru.flatten_parameters()

            models[f'ema_model_{fold}'] = ema_model.state_dict()

            checkpoint_weights = np.array([2 ** epoch for epoch in range(config.train_epochs)])
            checkpoint_weights = checkpoint_weights / checkpoint_weights.sum()

            valid_preds_fold = np.average(all_valid_preds, weights=checkpoint_weights, axis=0)
            valid_preds_fold[valid_nan_mask] = nan_pred
            auc_score, _ = evaluator.get_final_metric(valid_preds)
            print(f'cv model \t auc={auc_score:.5f}')

            ema_valid_preds_fold = eval_model(ema_model, valid_loader)
            ema_valid_preds_fold[valid_nan_mask] = nan_pred
            auc_score, _ = evaluator.get_final_metric(ema_valid_preds_fold)
            print(f'EMA model \t auc={auc_score:.5f}')

            train_preds[valid_idx] = valid_preds_fold
            ema_train_preds[valid_idx] = ema_valid_preds_fold

            test_preds_fold = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
            ema_test_preds_fold = eval_model(ema_model, test_loader)

            test_preds += test_preds_fold / config.n_splits
            ema_test_preds += ema_test_preds_fold / config.n_splits

    with timer('evaluate'):
        torch.save(models, MODEL_STATE)
        test_preds[test_nan_mask] = nan_pred
        ema_test_preds[test_nan_mask] = nan_pred
        evaluator = JigsawEvaluator(y_binary, y_identity_binary)
        auc_score, _ = evaluator.get_final_metric(train_preds)
        ema_auc_score, _ = evaluator.get_final_metric(ema_train_preds)
        print(f'cv score: {auc_score:<8.5f}')
        print(f'EMA cv score: {ema_auc_score:<8.5f}')

        train_preds = np.mean([train_preds, ema_train_preds], axis=0)
        test_preds = np.mean([test_preds, ema_test_preds], axis=0)
        auc_score, _ = evaluator.get_final_metric(train_preds)
        print(f'final prediction score: {auc_score:<8.5f}')

        if config.pseudo_label:
            test_preds = test_preds * 0.9 + test_y[:, 0] * 0.1

        submission = pd.DataFrame({
            'id': test['id'],
            'prediction': test_preds
        })
        submission.to_csv(SUBMISSION_PATH, index=False)


if __name__ == '__main__':
    main()
