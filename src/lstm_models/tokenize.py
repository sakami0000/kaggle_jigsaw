from collections import Counter


def build_vocab(texts, max_features):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {
        'token2id': {'<PAD>': 0, '<UNK>': max_features + 1},
        'id2token': {}
    }
    vocab['token2id'].update(
        {token: _id + 1 for _id, (token, count) in
         enumerate(counter.most_common(max_features))})
    vocab['id2token'] = {v: k for k, v in vocab['token2id'].items()}
    return vocab


def tokenize(texts, vocab, max_len):
    
    def text2ids(text, token2id):
        return [
            token2id.get(token, len(token2id) - 1)
            for token in text.split()[:max_len]]
    
    return [
        text2ids(text, vocab['token2id'])
        for text in texts]
