from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpatialDropout(nn.Dropout2d):

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class PretrainedEmbedding(nn.Module):

    def __init__(self, embedding_matrix):
        super(PretrainedEmbedding, self).__init__()
        embed_size = embedding_matrix.shape[1]
        max_features = embedding_matrix.shape[0]
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

    def forward(self, indices):
        return self.embedding(indices)


def init_weight(weight, method):
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def nn_init(nn_module, method='xavier'):
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)


class ProjSumEmbedding(nn.Module):

    def __init__(self, embedding_matrices: List[np.ndarray], output_size):
        super(ProjSumEmbedding, self).__init__()
        assert len(embedding_matrices) > 0

        self.embedding_count = len(embedding_matrices)
        self.output_size = output_size
        self.embedding_projectors = nn.ModuleList()
        for embedding_matrix in embedding_matrices:
            embedding_dim = embedding_matrix.shape[1]
            projection = nn.Linear(embedding_dim, self.output_size)
            nn_init(projection)

            self.embedding_projectors.append(nn.Sequential(
                PretrainedEmbedding(embedding_matrix),
                projection
            ))

    def forward(self, x):
        projected = [embedding_projector(x) for embedding_projector in self.embedding_projectors]
        return F.relu(sum(projected))


class Capsule(nn.Module):

    def __init__(self, input_dim_capsule=1024, num_capsule=5, dim_capsule=5, routings=4):
        super(Capsule, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.activation = self.squash
        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))

    def forward(self, x):
        u_hat_vecs = torch.matmul(x, self.W)
        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(
            0, 2, 1, 3).contiguous()  # (batch_size,num_capsule,input_num_capsule,dim_capsule)
        with torch.no_grad():
            b = torch.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            c = torch.nn.functional.softmax(b, dim=1)  # (batch_size,num_capsule,input_num_capsule)
            outputs = self.activation(torch.sum(c.unsqueeze(-1) * u_hat_vecs, dim=2))  # bij,bijk->bik
            if i < self.routings - 1:
                b = (torch.sum(outputs.unsqueeze(2) * u_hat_vecs, dim=-1))  # bik,bijk->bij
        return outputs  # (batch_size, num_capsule, dim_capsule)

    @staticmethod
    def squash(x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + 1e-7)
        return x / scale


class Attention(nn.Module):

    def __init__(self, feature_dim, maxlen=70):
        super().__init__()
        self.attention_fc = nn.Linear(feature_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1, maxlen, 1, requires_grad=True))

    def forward(self, rnn_output):
        """
        forward attention scores and attended vectors
        :param rnn_output: (#batch, #seq_len, #feature)
        :return: attended_outputs (#batch, #feature)
        """
        attention_weights = self.attention_fc(rnn_output)
        seq_len = rnn_output.size(1)
        attention_weights = self.bias[:, :seq_len, :] + attention_weights
        attention_weights = torch.tanh(attention_weights)
        attention_weights = torch.exp(attention_weights)
        attention_weights_sum = torch.sum(attention_weights, dim=1, keepdim=True) + 1e-7
        attention_weights = attention_weights / attention_weights_sum
        attended = torch.sum(attention_weights * rnn_output, dim=1)
        return attended
