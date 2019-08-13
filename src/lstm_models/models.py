import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import SpatialDropout, ProjSumEmbedding, Capsule, Attention


class LstmGruNet(nn.Module):

    def __init__(self, embedding_matrices, num_aux_targets, embedding_size=256, lstm_units=128,
                 gru_units=128):
        super(LstmGruNet, self).__init__()
        self.embedding = ProjSumEmbedding(embedding_matrices, embedding_size)
        self.embedding_dropout = SpatialDropout(0.2)

        self.lstm = nn.LSTM(embedding_size, lstm_units, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_units * 2, gru_units, bidirectional=True, batch_first=True)

        dense_hidden_units = gru_units * 4
        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)

        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h1, _ = self.lstm(h_embedding)
        h2, _ = self.gru(h1)

        # global average pooling
        avg_pool = torch.mean(h2, 1)
        # global max pooling
        max_pool, _ = torch.max(h2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


class LstmGruModel(nn.Module):

    def __init__(self, embedding_matrix: np.ndarray,
                 lstm_hidden_size=120, gru_hidden_size=60,
                 embedding_dropout=0.2, out_size=20, out_dropout=0.1):
        super(LstmGruModel, self).__init__()
        self.gru_hidden_size = gru_hidden_size

        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(embedding_dropout)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(gru_hidden_size * 6, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_dropout)
        self.out = nn.Linear(out_size, 1)
        
    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.apply_spatial_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, hh_gru = self.gru(h_lstm)

        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


class LstmCapsuleAttenModel(nn.Module):

    def __init__(self, embedding_matrix, maxlen=200, lstm_hidden_size=128, gru_hidden_size=128,
                 embedding_dropout=0.2, dropout1=0.2, dropout2=0.1, out_size=16,
                 num_capsule=5, dim_capsule=5, caps_out=1, caps_dropout=0.3):
        super(LstmCapsuleAttenModel, self).__init__()

        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(embedding_dropout)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)
        
        self.lstm_attention = Attention(lstm_hidden_size * 2, maxlen=maxlen)
        self.gru_attention = Attention(gru_hidden_size * 2, maxlen=maxlen)
        
        self.capsule = Capsule(input_dim_capsule=gru_hidden_size * 2,
                               num_capsule=num_capsule,
                               dim_capsule=dim_capsule)
        self.dropout_caps = nn.Dropout(caps_dropout)
        self.lin_caps = nn.Linear(num_capsule * dim_capsule, caps_out)

        self.norm = nn.LayerNorm(lstm_hidden_size * 2 + gru_hidden_size * 6 + caps_out)
        self.dropout1 = nn.Dropout(dropout1)
        self.linear = nn.Linear(lstm_hidden_size * 2 + gru_hidden_size * 6 + caps_out, out_size)
        self.dropout2 = nn.Dropout(dropout2)
        self.out = nn.Linear(out_size, 1)
        
    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.apply_spatial_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)
        
        content3 = self.capsule(h_gru)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)
        content3 = self.dropout_caps(content3)
        content3 = torch.relu(self.lin_caps(content3))

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, content3, avg_pool, max_pool), 1)
        conc = self.norm(conc)
        conc = self.dropout1(conc)
        conc = torch.relu(conc)
        conc = self.linear(conc)
        conc = self.dropout2(conc)
        out = self.out(conc)

        return out


class LstmConvModel(nn.Module):

    def __init__(self, embedding_matrix, lstm_hidden_size=128, gru_hidden_size=128, n_channels=64,
                 embedding_dropout=0.2, out_size=20, out_dropout=0.1):
        super(LstmConvModel, self).__init__()

        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)

        self.lstm = nn.LSTM(embedding_matrix.shape[1], lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(gru_hidden_size * 2, n_channels, 3, padding=2)
        nn.init.xavier_uniform_(self.conv.weight)

        self.linear = nn.Linear(n_channels * 2, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_dropout)
        self.out = nn.Linear(out_size, 1)

    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.apply_spatial_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        h_gru = h_gru.transpose(2, 1)
        conv = self.conv(h_gru)

        conv_avg_pool = torch.mean(conv, 2)
        conv_max_pool, _ = torch.max(conv, 2)

        conc = torch.cat((conv_avg_pool, conv_max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out
