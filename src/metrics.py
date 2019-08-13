import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from ..config.base import IDENTITY_COLUMNS


class JigsawEvaluator:

    def __init__(self, y_target: np.ndarray, y_identity: np.ndarray, power=-5, overall_model_weight=0.25):
        self.y = (y_target >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            # return np.nan
            return 1e-15

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bias_metrics_for_model(self, y_pred):
        metrics = np.zeros((3, self.n_subgroups))
        record = {
            'subgroup_auc': {},
            'bpsn_auc': {},
            'bnsp_auc': {}
        }
        for i in range(self.n_subgroups):
            metrics[0, i] = self._compute_subgroup_auc(i, y_pred)
            metrics[1, i] = self._compute_bpsn_auc(i, y_pred)
            metrics[2, i] = self._compute_bnsp_auc(i, y_pred)
            subgroup_name = IDENTITY_COLUMNS[i]
            record['subgroup_auc'][subgroup_name] = metrics[0, i]
            record['bpsn_auc'][subgroup_name] = metrics[1, i]
            record['bnsp_auc'][subgroup_name] = metrics[2, i]
        return metrics, record

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred: np.ndarray):
        bias_metrics, bias_record = self._compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_auc = self._calculate_overall_auc(y_pred)
        overall_score = self.overall_model_weight * overall_auc
        bias_score = (1 - self.overall_model_weight) * bias_score
        final_score = overall_score + bias_score

        bias_record['overall_auc'] = overall_auc
        bias_record['final_score'] = final_score
        bias_record['mean_subgroup_auc'] = self._power_mean(bias_metrics[0])
        bias_record['mean_bpsn_auc'] = self._power_mean(bias_metrics[1])
        bias_record['mean_bnsp_auc'] = self._power_mean(bias_metrics[2])
        return final_score, bias_record


def accuracy(ys, ps):
    return torch.mean(((ps >= 0.5) == (ys >= 0.5)).type(torch.FloatTensor)).item()
