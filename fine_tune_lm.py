from argparse import ArgumentParser
from pathlib import Path
import warnings

from apex import amp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config.base import (
    TOXICITY_COLUMN, IDENTITY_COLUMNS, AUX_TOXICITY_COLUMNS,
    OLD_TOXICITY_COLUMN, OLD_IDENTITY_COLUMNS, OLD_AUX_TOXICITY_COLUMNS,
    TRAIN_DATA, TEST_DATA, SAMPLE_SUBMISSION,
    TRAIN_OLD, TEST_OLD, SAMPLE_OLD)

from src.data import TextDataset, LengthBucketingDataLoader
from src.loss import CustomLoss
from src.metrics import JigsawEvaluator
from src.utils import timer, seed_torch, load_config
from src.weights import training_weights

from src.language_models.models import GPT2ClassificationHeadModel
from src.language_models.tokenize import MyTokenizer
from src.language_models.utils import get_optimizer_params, predict


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--valid', action='store_true')
    args = parser.parse_args()

    config_file = Path(args.config_file)
    config = load_config(config_file)

    config.setdefault('max_len', 220)
    config.setdefault('max_head_len', 128)
    config.setdefault('epochs', 2)
    config.setdefault('down_sample_frac', 0.5)
    config.setdefault('lr', 1.5e-5)
    config.setdefault('batch_size', 16)
    config.setdefault('accumulation_steps', 4)
    config.setdefault('lr_weight_decay_coef', 1.0)
    config.setdefault('warmup', 0.05)
    config.setdefault('old_data', False)
    config.setdefault('old_fine_tuned', False)
    config.setdefault('device', 'cuda')
    config.setdefault('seed', 1234)

    assert 'lm_model_name' in config
    assert not (config.old_fine_tuned and config.old_data)
    assert config.max_len >= config.max_head_len
    assert config.epochs <= 2

    lm_model_name = config_file.stem
    if config.old_fine_tuned:
        PRETRAINED_PATH = Path(f'../output/{lm_model_name}_old_fine_tune/')
        assert PRETRAINED_PATH.exists()
    else:
        PRETRAINED_PATH = args.lm_model
    MODE = args.lm_model[:4]
    LOWER_CASE = 'uncased' in args.lm_model
    LARGE_MODEL = 'large' in args.lm_model
    DEVICE = torch.device(config.device)

    if config.old_data:
        lm_model_name += '_old_fine_tune'

    if args.valid:
        valid_size = 200000
        shuffle_seed = 1029
        lm_model_name += '_valid'
    else:
        valid_size = 0
        shuffle_seed = config.seed

    OUT_DIR = Path(f'../output/{lm_model_name}/')
    TEST_SUBMISSION = OUT_DIR / 'submission.csv'
    VALID_SUBMISSION = OUT_DIR / 'valid_submission.csv'
    OUT_DIR.mkdir(exist_ok=True)

    warnings.filterwarnings('ignore')
    seed_torch(config.seed)

    if not args.old:
        train_data = TRAIN_DATA
        test_data = TEST_DATA
        sample_submission = SAMPLE_SUBMISSION
        train_size = 1804874 - valid_size
    else:
        train_data = TRAIN_OLD
        test_data = TEST_OLD
        sample_submission = SAMPLE_OLD
        train_size = 159571 - valid_size

        TOXICITY_COLUMN = OLD_TOXICITY_COLUMN
        IDENTITY_COLUMNS = OLD_IDENTITY_COLUMNS
        AUX_TOXICITY_COLUMNS = OLD_AUX_TOXICITY_COLUMNS

    if MODE == 'bert':
        from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam

        lm_tokenizer = BertTokenizer.from_pretrained(
            args.lm_model, cache_dir=None, do_lower_case=LOWER_CASE)
        model = BertForSequenceClassification.from_pretrained(
            PRETRAINED_PATH, cache_dir=None, num_labels=1 + len(AUX_TOXICITY_COLUMNS))
        optimizer_class = BertAdam
    else:
        from pytorch_pretrained_bert import GPT2Tokenizer, OpenAIAdam, GPT2Model

        lm_tokenizer = GPT2Tokenizer.from_pretrained(args.lm_model, cache_dir=None)
        model = GPT2ClassificationHeadModel.from_pretrained(PRETRAINED_PATH,
                                                            clf_dropout=config.get('dropout_rate', 0.1),
                                                            n_class=1 + len(AUX_TOXICITY_COLUMNS))
        optimizer_class = OpenAIAdam
        assert config.lr_weight_decay_coef == 1.0

    with timer('preprocess'):
        tokenizer = MyTokenizer(lm_tokenizer, config.max_len, config.max_head_len, MODE)
        df_train = pd.read_csv(TRAIN_DATA).sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        df_train['comment_text'] = df_train['comment_text'].astype(str)
        df_train = df_train.fillna(0)
        X_train = tokenizer.tokenize(
            df_train['comment_text'].fillna('DUMMY_VALUE'), num_threads=16, chunksize=5000)

        df_test = pd.read_csv(TEST_DATA)
        df_test['comment_text'] = df_test['comment_text'].astype(str)
        df_test = df_test.fillna(0)
        X_test = tokenizer.tokenize(
            df_test['comment_text'].fillna('DUMMY_VALUE'), num_threads=16, chunksize=5000)

        df_train.drop(['comment_text'], axis=1, inplace=True)
        df_test.drop(['comment_text'], axis=1, inplace=True)

        X_valid = X_train[train_size:]
        X_train = X_train[:train_size]

        y_identity_train = df_train[IDENTITY_COLUMNS].values
        y_annotator_counts_train = df_train['toxicity_annotator_count'].values

        weights = training_weights(df_train, TOXICITY_COLUMN, IDENTITY_COLUMNS)
        y_train = np.hstack((
            df_train[TOXICITY_COLUMN].values.reshape(-1, 1),
            weights.reshape(-1, 1),
            df_train[AUX_TOXICITY_COLUMNS].values
        ))

        y_valid = y_train[train_size:]
        y_train = y_train[:train_size]
        y_identity_valid = y_identity_train[train_size:]
        y_identity_train = y_identity_train[:train_size]
        y_annotator_counts_valid = y_annotator_counts_train[train_size:]
        y_annotator_counts_train = y_annotator_counts_train[:train_size]
        loss_weight = 1.0 / weights.mean() if not args.old else None

        # drop negative samples here
        frac = config.down_sample_frac
        target_negative = (y_train > 0.0).sum(axis=1) == 1
        identity_negative = (y_identity_train > 0.0).sum(axis=1) == 0
        negative_mask = identity_negative & target_negative
        negative_indices = np.arange(len(y_train))[negative_mask]
        drop_indices_0 = set(negative_indices[:int(len(negative_indices) * frac)])
        drop_indices_1 = set(negative_indices[int(len(negative_indices) * (1 - frac)):])
        drop_indices_list = [drop_indices_0, drop_indices_1]

        len_train = len(y_train) - len(drop_indices_0)

    with timer('train'):
        model.zero_grad()
        model = model.to(DEVICE)
        num_layers = 24 if LARGE_MODEL else 12
        optimizer_grouped_parameters = get_optimizer_params(
            model, config.lr, config.lr_weight_decay_coef, num_layers)
        num_train_optimization_steps = int(
            config.epochs * len_train / config.batch_size / config.accumulation_steps)

        optimizer = optimizer_class(optimizer_grouped_parameters,
                                    lr=config.lr,
                                    warmup=config.warmup,
                                    t_total=num_train_optimization_steps)

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        model = model.train()

        batch_count = len_train // config.batch_size
        loss_fn = CustomLoss(loss_weight)
        for epoch, drop_indices in zip(range(config.epochs), drop_indices_list):
            sample_indices = np.array([i for i in range(len(y_train)) if i not in drop_indices])
            X_sampled_train = [X_train[i] for i in sample_indices]
            y_sampled_train = y_train[sample_indices]
            y_sampled_identity_train = y_identity_train[sample_indices]
            y_sampled_annotator_counts_train = y_annotator_counts_train[sample_indices]
            train_dataset = TextDataset(X_sampled_train, y_sampled_train,
                                        y_sampled_identity_train, y_sampled_annotator_counts_train)
            train_loader = LengthBucketingDataLoader(
                train_dataset, shuffle=True, drop_last=True, batch_size=config.batch_size)
            tk0 = tqdm(enumerate(train_loader), total=batch_count)
            optimizer.zero_grad()
            for i, (x_batch, _, a_batch, y_batch, y_identity_batch) in tk0:
                y_pred = model(x_batch.to(DEVICE), attention_mask=(x_batch > 0).to(DEVICE), labels=None)
                loss = loss_fn(y_pred, y_batch.to(DEVICE))
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (i + 1) % config.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        model.save_pretrained(OUT_DIR)

    with timer('evaluate'):
        if args.valid:
            valid_dataset = TextDataset(X_valid, y_valid, y_identity_valid, y_annotator_counts_valid)
            valid_preds = predict(model, valid_dataset, device=DEVICE)

            df_valid = df_train.tail(valid_size)
            df_valid['model1'] = valid_preds
            evaluator = JigsawEvaluator(df_valid[TOXICITY_COLUMN].values, df_valid[IDENTITY_COLUMNS].values)
            final_score, _ = evaluator.get_final_metric(df_valid['model1'].values)

            valid_prediction = predict(model, TextDataset(X_valid), device=DEVICE)
            valid_submission = pd.DataFrame({
                'id': df_valid['id'],
                'prediction': valid_prediction 
            })
            valid_submission.to_csv(VALID_SUBMISSION, index=False)
            print(f'validation score: {final_score:.5f}')

        test_prediction = predict(model, TextDataset(X_test), device=DEVICE)
        submission = pd.DataFrame({
            'id': df_test['id'],
            'prediction': test_prediction 
        })
        submission.to_csv(TEST_SUBMISSION, index=False)


if __name__ == '__main__':
    main()
