import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class TextDataset(Dataset):

    def __init__(self, token_lists: List[List[int]], targets: np.ndarray = None, identities: np.ndarray = None,
                 annotator_counts: np.ndarray = None):
        assert targets is None or type(targets) == np.ndarray
        assert identities is None or type(identities) == np.ndarray
        super(TextDataset, self).__init__()
        self.token_lists = token_lists
        self.targets = targets
        self.identities = identities
        self.annotator_counts = annotator_counts

    def __len__(self) -> int:
        return len(self.token_lists)

    def __getitem__(self, item):
        if self.targets is None:
            return self.token_lists[item], item
        return self.token_lists[item], item, self.annotator_counts[item], self.targets[item], self.identities[item]

    def collate_fn(self, batch):
        transposed = list(zip(*batch))
        max_len = max([len(x) for x in transposed[0]])
        tokens = np.zeros((len(batch), max_len), dtype=np.int64)
        for i, row in enumerate(transposed[0]):
            row = np.array(row[:min(max_len, len(row))])
            tokens[i, :len(row)] = row

        # token_lists, indices
        tensors = [
            torch.from_numpy(tokens),
            torch.Tensor(transposed[1]).type(torch.IntTensor),
        ]
        for i in range(2, len(transposed)):
            tensors.append(torch.Tensor(transposed[i]).type(torch.FloatTensor))
        return tensors


class LengthBucketingDataLoader(object):

    def __init__(self, dataset: TextDataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        self.large_bucket_loader = DataLoader(dataset=dataset, batch_size=batch_size * 100, shuffle=shuffle,
                                              sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                                              collate_fn=self.nop_collate_fn, pin_memory=pin_memory, drop_last=False,
                                              timeout=timeout, worker_init_fn=worker_init_fn)
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.collate_fn = dataset.collate_fn

    @staticmethod
    def nop_collate_fn(batch):
        return batch

    def __iter__(self):
        for large_batch in self.large_bucket_loader:
            assert type(large_batch[0])
            large_batch = sorted(large_batch, key=lambda example: len(example[0]))

            small_batches = []
            for start_idx in range(0, len(large_batch), self.batch_size):
                end_idx = min(len(large_batch), start_idx + self.batch_size)
                small_batch = large_batch[start_idx:end_idx]
                if end_idx - start_idx == self.batch_size or not self.drop_last:
                    small_batches.append(self.collate_fn(small_batch))
            random.shuffle(small_batches)

            for small_batch in small_batches:
                yield small_batch


class TokenDataset(Dataset):

    def __init__(self, seqs, targets=None, maxlen=200):
        if targets is not None:
            self.targets = targets
        else:
            self.targets = np.random.randint(2, size=(len(seqs),))
        
        self.seqs = seqs
        self.maxlen = maxlen
        
    def __len__(self):
        return len(self.seqs)
        
    def get_keys(self):
        lens = np.fromiter(
            ((min(self.maxlen, len(seq))) for seq in self.seqs),
            dtype=np.int32)
        return lens
        
    def __getitem__(self, index):
        return index, self.seqs[index], self.targets[index]


def collate_fn(data):

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        max_len = max(lens)

        padded_seqs = torch.zeros(len(seqs), max_len).long()
        for i, seq in enumerate(seqs):
            start = max_len - lens[i]
            padded_seqs[i, start:] = torch.LongTensor(seq)
        return padded_seqs

    index, seqs, targets = zip(*data)
    seqs = _pad_sequences(seqs)
    return index, seqs, torch.FloatTensor(targets)


class BucketSampler(Sampler):

    def __init__(self, data_source, sort_keys, bucket_size=None, batch_size=1048, shuffle_data=True):
        super().__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_keys = sort_keys
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_keys)
        self.weights = None

        if not shuffle_data:
            self.index = self.prepare_buckets()
        else:
            self.index = None

    def set_weights(self, weights):
        assert weights >= 0
        total = np.sum(weights)
        if total != 1:
            weights = weights / total
        self.weights = weights

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_keys)
            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = self.prepare_buckets(indices)
        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_keys)
        
    def prepare_buckets(self, indices=None):
        lens = - self.sort_keys
        assert self.bucket_size % self.batch_size == 0 or self.bucket_size == len(lens)

        if indices is None:
            if self.shuffle:
                indices = shuffle(np.arange(len(lens), dtype=np.int32))
                lens = lens[indices]
            else:
                indices = np.arange(len(lens), dtype=np.int32)

        #  bucket iterator
        def divide_chunks(l, n):
            if n == len(l):
                yield np.arange(len(l), dtype=np.int32), l
            else:
                # looping till length l
                for i in range(0, len(l), n):
                    data = l[i:i + n]
                    yield np.arange(i, i + len(data), dtype=np.int32), data

        new_indices = []
        extra_batch = None
        for chunk_index, chunk in divide_chunks(lens, self.bucket_size):
            # sort indices in bucket by descending order of length
            indices_sorted = chunk_index[np.argsort(chunk, axis=-1)]
            batches = []
            for _, batch in divide_chunks(indices_sorted, self.batch_size):
                if len(batch) == self.batch_size:
                    batches.append(batch.tolist())
                else:
                    assert extra_batch is None
                    assert batch is not None
                    extra_batch = batch
    
            # shuffling batches within buckets
            if self.shuffle:
                batches = shuffle(batches)
            for batch in batches:
                new_indices.extend(batch)
    
        if extra_batch is not None:
            new_indices.extend(extra_batch)
        return indices[new_indices]
