from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from config.base import TOXICITY_COLUMN, IDENTITY_COLUMNS
from src.metrics import JigsawEvaluator
from src.utils import load_config

OUT_DIR = Path('../output/')
VALID_DIR = OUT_DIR.glob('*_valid/')
models = [path.stem[:-6] for path in VALID_DIR]


def objective(trial, train_fold, evaluator):
    params = {
        model: trial.suggest_uniform(model, 0.0, 1.0)
        for model in models
    }
    train_fold = np.array(train_fold)
    train_preds = np.average(train_fold, weights=params.values(), axis=1)
    score, _ = evaluator.get_final_metric(train_preds)
    return 1 - score


def main():
    config = load_config('./config/blend.json')
    config.setdefault('n_folds', 10)
    config.setdefault('n_trials', 300)
    config.setdefault('threshold', 0.03)

    df_valid = pd.concat([
        pd.read_csv(path / 'valid_submission.csv', index_col='id')
        for path in VALID_DIR
    ], axis=0)
    train_scores = []
    valid_scores = []
    params = {model: [] for model in models}

    for i in range(config.n_folds):
        df_valid = df_valid.sample(frac=1, random_state=i).reset_index(drop=True)
        train_fold = df_valid[:len(df_valid) // 2]
        valid_fold = df_valid[len(df_valid) // 2:]
        train_evaluator = JigsawEvaluator(
            train_fold[TOXICITY_COLUMN].values, train_fold[IDENTITY_COLUMNS].values)
        valid_evaluator = JigsawEvaluator(
            valid_fold[TOXICITY_COLUMN].values, valid_fold[IDENTITY_COLUMNS].values)
        
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, train_fold.values, train_evaluator),
                       n_trials=config.n_trials)
        trial = study.best_trial
        train_scores.append(1 - trial.value)
        values = np.array(list(trial.params.values()))
        values /= values.sum()
        for key, value in zip(trial.params.keys(), values):
            params[key].append(value)
        
        valid_preds = np.zeros((len(valid_fold)))
        for key, value in trial.params.items():
            valid_preds += valid_fold[key].values * value
        score, _ = valid_evaluator.get_final_metric(valid_preds)
        valid_scores.append(score)

    for i, (train_score, valid_score) in enumerate(zip(train_scores, valid_scores)):
        print(f'fold {str(i + 1):2s} - train: {train_score:.5f}, valid: {valid_score:.5f}')

    print('-' * 20)
    print(f'train mean: {np.mean(train_scores):.5f}, var: {np.var(train_scores):.7f}')
    print(f'valid mean: {np.mean(valid_scores):.5f}, var: {np.var(valid_scores):.7f}')

    print('-' * 20)
    for key, values in params.items():
        print(f'{key:25s} {np.mean(values):.6f} {np.var(values):.6f}')

    print('-' * 20)
    print(f'robust folds: threshold {config.threshold}')
    robust_folds = []
    robust_train_scores = []
    robust_valid_scores = []
    for i, (train_score, valid_score) in enumerate(zip(train_scores, valid_scores)):
        if np.abs(train_score - valid_score) < config.threshold:
            robust_folds.append(i)
            robust_train_scores.append(train_score)
            robust_valid_scores.append(valid_score)
        print(' '.join(map(str, robust_folds)))

    print('-' * 20)
    print(f'train mean: {np.mean(robust_train_scores):.5f}, var: {np.var(robust_train_scores):.7f}')
    print(f'valid mean: {np.mean(robust_valid_scores):.5f}, var: {np.var(robust_valid_scores):.7f}')

    print('-' * 20)
    for key, values in params.items():
        robust_values = np.array(values)[robust_folds]
        print(f'{key:25s} {np.mean(robust_values):.6f} {np.var(robust_values):.6f}')    


if __name__ == '__main__':
    main()
