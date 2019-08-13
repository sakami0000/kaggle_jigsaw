from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import torch

from config.base import (
    TOXICITY_COLUMN, IDENTITY_COLUMNS, AUX_TOXICITY_COLUMNS,
    EMBEDDING_FASTTEXT, EMBEDDING_GLOVE,
    TRAIN_DATA, TEST_DATA, SAMPLE_SUBMISSION)

from src.data import TextDataset
from src.loss import CustomLoss
from src.utils import timer, seed_torch, load_config
from src.weights import training_weights

from src.lstm_models.load_data import load_embedding
from src.lstm_models.models import LstmGruNet
from src.lstm_models.preprocess import preprocess
from src.lstm_models.utils import train, predict, display_tables
from src.lstm_models.tokenize import build_vocab, tokenize


def main():
    parser = ArgumentParser()
    parser.add_argument('--valid', action='store_true')
    args = parser.parse_args()

    config = load_config('./config/lstm_f.json')
    config.setdefault('max_len', 220)
    config.setdefault('max_features', 100000)
    config.setdefault('batch_size', 512)
    config.setdefault('train_epochs', 10)
    config.setdefault('tolerance', 10)
    config.setdefault('num_folds', 5)
    config.setdefault('lr', 1e-3)
    config.setdefault('loss_alpha', 0.1)
    config.setdefault('loss_beta', 1.0)
    config.setdefault('device', 'cuda')
    config.setdefault('seed', 1029)

    device = torch.device(config.device)

    OUT_DIR = Path(f'../output/lstm_f/')
    submission_file_name = 'valid_submission.csv' if args.valid else 'submission.csv'
    SUBMISSION_PATH = OUT_DIR / submission_file_name
    OUT_DIR.mkdir(exist_ok=True)

    warnings.filterwarnings('ignore')
    seed_torch(config.seed)

    with timer('preprocess'):
        train = pd.read_csv(TRAIN_DATA)
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
        X_train = train['comment_text'].fillna('_##_').values
        X_test = test['comment_text'].fillna('_##_').values
        
        # get the target values
        weights = training_weights(train, TOXICITY_COLUMN, IDENTITY_COLUMNS)
        loss_weight = 1.0 / weights.mean()
        y_train_identity = train[IDENTITY_COLUMNS].values
        y_train_annotator_counts = train['toxicity_annotator_count'].values
        y_train = np.hstack((
            train[TOXICITY_COLUMN].values.reshape(-1, 1),
            weights.reshape(-1, 1),
            train[AUX_TOXICITY_COLUMNS].values
        ))

        train_nan_mask = X_train == '_##_'
        test_nan_mask = X_test == '_##_'

        vocab = build_vocab(chain(X_train), config.max_features)
        fasttext_embedding_matrix = load_embedding(EMBEDDING_FASTTEXT, vocab['token2id'])
        glove_embedding_matrix = load_embedding(EMBEDDING_GLOVE, vocab['token2id'])

        joblib.dump(vocab, OUT_DIR / 'vocab.pkl')
        np.save(OUT_DIR / 'fasttext_embedding_matrix', fasttext_embedding_matrix)
        np.save(OUT_DIR / 'glove_embedding_matrix', glove_embedding_matrix)

        X_train = np.array(tokenize(X_train, vocab, config.max_len))
        X_test = np.array(tokenize(X_test, vocab, config.max_len))

        all_related_columns = [TOXICITY_COLUMN] + AUX_TOXICITY_COLUMNS + IDENTITY_COLUMNS
        negative_indices = np.arange(0, len(train))[
            (train[all_related_columns] == 0.0).sum(axis=1) == len(all_related_columns)]

    with timer('train'):
        skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=1)
        num_aux_targets = y_train.shape[-1] - 2
        custom_loss = CustomLoss(
            loss_weight, alpha=config.loss_alpha, beta=config.loss_beta,
            use_annotator_counts=True,
            weight_from_annotator_counts=lambda x: torch.log(x + 2))
        test_dataset = TextDataset(token_lists=X_test)
        test_prediction = np.zeros(len(test_dataset))
        test_prediction_count = 0
        models = {}
        for i, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train[:, 0] >= 0.5)):
            seed_torch(i)
            np.random.shuffle(negative_indices)
            drop_indices = set(negative_indices[:len(negative_indices) // 2])
            train_idx = [i for i in train_idx if i not in drop_indices]
            train_token_lists = [X_train[i] for i in train_idx]
            valid_token_lists = [X_train[i] for i in valid_idx]
            train_dataset = TextDataset(token_lists=train_token_lists, targets=y_train[train_idx],
                                        identities=y_train_identity[train_idx],
                                        annotator_counts=y_train_annotator_counts[train_idx])
            valid_dataset = TextDataset(token_lists=valid_token_lists, targets=y_train[valid_idx],
                                        identities=y_train_identity[valid_idx],
                                        annotator_counts=y_train_annotator_counts[valid_idx])
            model = LstmGruNet(embedding_matrices=[glove_embedding_matrix, fasttext_embedding_matrix],
                               num_aux_targets=num_aux_targets).to(device)
            model, records = train(model, custom_loss, train_dataset, valid_dataset, device=device,
                                   batch_size=config.batch_size, num_epochs=config.train_epochs,
                                   tolerance=config.tolerance, lr=config.lr)
            test_prediction += predict(model, test_dataset, device)
            test_prediction_count += 1
            torch.save(model.state_dict(), OUT_DIR / f'model.{i}.json')

            with open(OUT_DIR / f'records.{i}.json', 'w') as f:
                import json
                json.dump(records, f, indent=4)

            submission = pd.DataFrame({
                'id': test['id'],
                'prediction': test_prediction / test_prediction_count
            })
            submission.to_csv(SUBMISSION_PATH, index=False)
            display_tables(OUT_DIR)


if __name__ == '__main__':
    main()
