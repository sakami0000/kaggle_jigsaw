from multiprocessing.pool import Pool
from typing import List, TypeVar

from pytorch_pretrained_bert import PreTrainedTokenizer
from tqdm import tqdm

Tokenizer = TypeVar('Tokenizer', bound=PreTrainedTokenizer)


class MyTokenizer:

    def __init__(self, tokenizer: Tokenizer, max_len=220, max_head_len=128, mode='bert'):
        assert max_len >= max_head_len
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_head_len = max_head_len
        self.mode = mode
        if self.mode == 'bert':
            self.max_len -= 2

    def _tokenize_one(self, text: str) -> List[int]:
        """
        when the sentence is longer then `max_len`,
        the first `max_head_len` and the last `max_len` - `max_head_len` words will 
        be used to train or inference
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_head_len] + tokens[self.max_head_len - self.max_len:]
        if self.mode == 'bert':
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, examples: List[str], num_threads=1, chunksize=1000):
        if num_threads < 1:
            raise ValueError('num_threads must be positive integer.')
        all_tokens = []
        total = len(examples)
        if num_threads == 1:
            for _, text in tqdm(enumerate(examples), total=total):
                all_tokens.append(self._tokenize_one(text))
        else:
            with Pool(num_threads) as pool:
                for _, tokens in tqdm(enumerate(pool.imap(self._tokenize_one, examples, chunksize=chunksize)),
                                      total=total):
                    all_tokens.append(tokens)
        return all_tokens
