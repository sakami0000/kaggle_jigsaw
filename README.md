## Jigsaw Unintended Bias in Toxicity Classification

3rd place solution by F.H.S.D.Y. of Jigsaw Unintended Bias in Toxicity Classification (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

Please see https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471#latest-582610 for more information.

### Requirements

```
apex
attrdict==2.0.1
nltk==3.4.4
numpy==1.16.4
optuna==0.13.0
pandas==0.24.2
pytorch-pretrained-bert==0.6.2
scikit-learn==0.21.2
torch==1.1.0
tqdm==4.32.1
```

apex from https://github.com/NVIDIA/apex

If you use BERT, you have to install  `pytorch-pretrained-bert` from pip.  
`pip install pytorch-pretrained-bert`

If you use GPT2, you have to install `pytorch-pretrained-bert` from git.  
`pip install git+https://github.com/pronkinnikita/pytorch-pretrained-BERT`

### Configuration

Hyper-parameters are managed by JSON files in `config` directory.

#### BERT & GPT2 configuration

- `lm_model_name`: required, type = str. Which language model to use.
- `max_len`: default = `220`, type = int. Maximum length of tokens used as input. In case of BERT, this contains `[CLS]` and `[SEP]`.
- `max_head_len`: default = `128`, type = int. Maximum length of first tokens used as input. This doesn't contain `[CLS]` or `[SEP]`. For example, when `max_len` = 220, `max_head_len` = 128, and using BERT, the input will be first 128 tokens and last 90 tokens.
- `epochs`: default = `2`, type = int. Training epochs. This must be 2 or less.
- `down_sample_frac`: default = `0.5`, type = float. Rate of dropped sample when negative down sampling.
- `lr`: default = `1.5e-5`, type = float. Learning rate.
- `batch_size`: deafult = `16`, type = int. Batch size.
- `accumulation_steps`: default = `4`, type = int. `gradient_accumulation_steps` in `pytorch_pretrained_bert`.
- `warmup`: default = `0.05`, type = float. Learning rate linearly increases from 0 to `lr` over `warmup` rate of training steps, and linearly decreases from `lr` to 0 over the rest steps.
- `old_data`: default = `false`, type = bool. Whether you use [old toxic competition data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) as training data or not.
- `old_fine_tuned`: default = `false`, type = bool. Set `true` if you use further fine-tuned weight with old toxic competition data as pre-trained weight, `false` otherwise.
- `device`: default = `cuda`, type = str, options: `cuda`, `cpu`. Device used for running.
- `seed`: default = `1234`, type = int. The desired seed.
- `dropout_rate`: default = `0.1`, type = float. Dropout rate for GPT2.

#### LSTM f configuration

- `max_len`: default = `220`, type = int. Maximum length of tokens used as input.
- `max_features`: default = `100000`, type = int. Maximum number of tokens used as input overall.
- `batch_size`: default = `512`, type = int. Batch size.
- `train_epochs`: default = `10`, type = int. Training epochs.
- `tolerance`: default = `10`, type = int. When score does not improve over `tolerance` epochs, training is aborted.
- `num_folds`: default = `5`, type = int. Number of folds of cross validation.
- `lr`: default = `1e-3`, type = float. Learning rate.
- `loss_alpha`: default = `0.1`, type = float. Coefficient for training weight.
- `loss_beta`: default = `1.0`, type = float. Coefficient for training weight.
- `device`: default = `cuda`, type = str, options: `cuda`, `cpu`. Device used for running.
- `seed`: default = `1234`, type = int. The desired seed.

#### LSTM s configuration

- `max_len`: default = `220`, type = int. Maximum length of tokens used as input.
- `max_features`: default = `100000`, type = int. Maximum number of tokens used as input overall.
- `batch_size`: default = `512`, type = int. Batch size.
- `train_epochs`: default = `6`, type = int. Training epochs.
- `n_splits`: default = `5`, type = int. Number of folds of cross validation.
- `start_lr`: default = `1e-4`, type = float. Initial learning rate of training.
- `max_lr`: default = `5e-3`, type = float. Maximum learning rate of training.
- `last_lr`: default = `1e-3`, type = float. Last learning rate of traiing.
- `warmup`: default = `0.2`, type = float. Learning rate increases from `start_lr` to `max_lr` over `warmup` rate of training steps following a cosine curve, and decreases from `max_lr` to `last_lr` over rest steps following a cosine curve.
- `pseudo_label`: default = `true`, type = bool. Whether you use pseudo labeling for training or not.
- `mu`: default = `0.9`, type = float. Rate of new weights in EMA.
- `updates_per_epoch`: default = `10`, type = int. How many times you update weights in EMA.
- `lstm_gru`: default = `{}`, type = dict. Hyper-parameters used in LSTM-GRU model.
- `lstm_capsule_atten`: default = `{}`, type = dict. Hyper-parameters used in LSTM-Capsule-Attention model.
- `lstm_conv`: default = `{}`, type = dict. Hyper-parameters used in LSTM-Conv model.
- `device`: default = `cuda`, type = str, options: `cuda`, `cpu`. Device used for running.
- `seed`: default = `1234`, type = int. The desired seed.

#### Computing blending weights configuration

- `n_folds`: default = 10, type = int. How many times you split validation data and run optuna.
- `n_trials`: default = 300, type = int. Number of trials of each running.
- `threshold`: default = 0.03, type = float. Acceptable error between train and valid score.

### Execution

#### BERT & GPT2 execution

You need to specify which configuration JSON file to use.

```
$ python fine_tune_lm.py \
    --config_file ./config/bert_large_cased.json
```

If you want to use further fine-tuned weights with old toxic competition data as pre-trained weights, first you need to run the script of further fine-tuning.

```
$ python fune_tune_lm.py \
    --config_file ./config/bert_base_cased_old_fine_tune.json
```

And then run the script of main fine-tuning.

```
$ python fine_tune_lm.py \
    --config_file ./config/bert_base_cased.json
```

#### LSTM f execution

You can run the script using

```
$ python train_lstm_f.py
```

#### LSTM s execution

You need to specify which model to use.

```
$ python train_lstm_s.py \
    --lstm_model lstm_gru
```

You can choose from

- `lstm_gru`
- `lstm_capsule_atten`
- `lstm_conv`

#### Computing blending weights execution

First you need to train models and predict labels of validation data with argument `--valid`

```
$ python fine_tune_lm.py \
    --config_file ./config/bert_large_cased.json
    --valid
```

Then run a command

```
$ python compute_blending_weights.py
```
