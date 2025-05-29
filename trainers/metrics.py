from typing import Union

import torch

# from ogb.graphproppred import Evaluator
# from ogb.lsc import PCQM4MEvaluator
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    auc,
    precision_recall_curve,
    recall_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
)

# from datasets.geom_drugs_dataset import GEOMDrugs
# from datasets.qm9_dataset import QM9Dataset


class CpError(nn.Module):
    def __init__(
        self,
        q=2,
        # range_vals=torch.arange(0.5, 11.5, 1),
        class_num=11,
    ):
        super(CpError, self).__init__()
        # self.smax = smax  # Softmax function or method
        self.q = q
        interval_len = 10 / (class_num - 1)
        self.range_vals = torch.arange(
            interval_len * 0.5, 10 + interval_len * 1.5, interval_len
        )  # Reference values for the categories (e.g., quantiles)

    def forward(self, probs, targets):
        # targets = torch.where(targets > 10.0, torch.tensor(10.0), targets)
        targets = torch.clamp(targets, max=10.0)
        # print(self.range_vals)

        pred = torch.sum(
            probs
            * self.range_vals.view(1, -1).expand(len(targets), -1).to(targets.device),
            dim=-1,
        )

        cp_error = torch.pow(pred - targets, self.q).mean()

        # Final loss calculation
        return cp_error


def percentile_excluding_index(vector, percentile):
    percentile_value = torch.quantile(vector, percentile)

    return percentile_value


def find_intervals_above_value_with_interpolation(x_values, y_values, cutoff):
    intervals = []
    start_x = None
    if y_values[0] >= cutoff:
        start_x = x_values[0]
    for i in range(len(x_values) - 1):
        x1, x2 = x_values[i], x_values[i + 1]
        y1, y2 = y_values[i], y_values[i + 1]

        if min(y1, y2) <= cutoff < max(y1, y2):
            # Calculate the x-coordinate where the line crosses the cutoff value
            x_cross = x1 + (x2 - x1) * (cutoff - y1) / (y2 - y1)

            if x1 <= x_cross <= x2:
                if start_x is None:
                    start_x = x_cross
                else:
                    intervals.append((start_x, x_cross))
                    start_x = None

    # If the line ends above cutoff, add the last interval
    if start_x is not None:
        intervals.append((start_x, x_values[-1]))

    return intervals


def calc_coverages_and_lengths(all_intervals, y):
    coverages = []
    lengths = []
    for idx, intervals in enumerate(all_intervals):
        if len(intervals) == 0:
            length = 0
            cov_val = 0
        else:
            length = 0
            cov_val = 0
            for interval in intervals:
                length += interval[1] - interval[0]
                if interval[1] >= y[idx].item() and y[idx].item() >= interval[0]:
                    cov_val = 1
        coverages.append(cov_val)
        lengths.append(length)

    return coverages, lengths


class coverages_and_lengths(nn.Module):
    def __init__(
        self,
        q=2,
        # range_vals=torch.arange(0.5, 11.5, 1),
        alpha=0.2,
        class_num=11,
    ):
        super(coverages_and_lengths, self).__init__()
        # self.smax = smax  # Softmax function or method
        self.q = q
        interval_len = 10 / (class_num - 1)
        self.range_vals = torch.arange(
            interval_len * 0.5, 10 + interval_len * 1.5, interval_len
        )  # Reference values for the categories (e.g., quantiles)
        self.alpha = alpha

    def forward(self, val_pred, targets, test_pred=None):
        # get_cp_lists
        # get_all_scores
        step_val = (max(self.range_vals) - min(self.range_vals)) / (
            len(self.range_vals) - 1
        )
        indices_up = torch.ceil((targets - min(self.range_vals)) / step_val).squeeze()
        indices_down = torch.floor(
            (targets - min(self.range_vals)) / step_val
        ).squeeze()

        how_much_each_direction = (
            targets.squeeze() - min(self.range_vals)
        ) / step_val - indices_down

        weight_up = how_much_each_direction
        weight_down = 1 - how_much_each_direction

        bad_indices = torch.where(
            torch.logical_or(
                targets.squeeze() > max(self.range_vals),
                targets.squeeze() < min(self.range_vals),
            )
        )
        indices_up[bad_indices] = 0
        indices_down[bad_indices] = 0

        scores = val_pred

        all_scores = (
            scores[torch.arange(len(val_pred)), indices_up.long()] * weight_up
            + scores[torch.arange(len(val_pred)), indices_down.long()] * weight_down
        )
        all_scores[bad_indices] = 0

        # end

        pred_scores = test_pred if test_pred is not None else val_pred
        percentile_val = percentile_excluding_index(all_scores, self.alpha)
        # targets = torch.where(targets > 10.0, torch.tensor(10.0), targets)
        all_intervals = []
        for i in range(len(pred_scores)):
            all_intervals.append(
                find_intervals_above_value_with_interpolation(
                    self.range_vals, pred_scores[i], percentile_val
                )
            )
        # end

        actual_intervals = self.invert_intervals(all_intervals)
        return calc_coverages_and_lengths(actual_intervals, targets)


class F1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        try:
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                targets_indices = torch.argmax(targets, dim=1)
                pred_classes = torch.argmax(preds, dim=1)
            else:
                pred_classes = np.around(preds.squeeze().cpu())
                targets_indices = targets.squeeze().cpu()

            score = f1_score(targets_indices.cpu(), pred_classes.cpu())
            return score
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, handle the error in a specific way,
            # like returning a default score or re-raising the exception.
            # For now, let's return a default score of 0.
            return 0


def mcc_score(predict_proba, label):
    trans_pred = np.ones(predict_proba.shape)
    trans_label = np.ones(label.shape)
    trans_pred[predict_proba < 0.5] = -1
    trans_label[label != 1] = -1
    # print(trans_pred.shape, trans_pred)
    # print(trans_label.shape, trans_label)
    mcc = matthews_corrcoef(trans_label, trans_pred)
    # mcc = metricser.matthews_corrcoef(trans_pred, trans_label)
    return mcc


class MCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        try:
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                targets_indices = torch.argmax(targets, dim=1)
                pred_classes = torch.argmax(preds, dim=1)
            else:
                score = mcc_score(preds.squeeze().cpu(), targets.squeeze().cpu())
                return score
            score = matthews_corrcoef(targets_indices.cpu(), pred_classes.cpu())
            return score
        except Exception as e:
            print(f"An error occurred: {e}")
            # print(targets_indices.shape,pred_classes.shape)
            # Optionally, handle the error in a specific way,
            # like returning a default score or re-raising the exception.
            # For now, let's return a default score of 0.
            return 0


class ROCAUC(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets_indices = torch.argmax(targets, dim=1)
            mask = targets == 1
            pred_score = preds[mask]
            pred_score = pred_score.view(-1)
        else:
            pred_score = preds.squeeze().view(-1)
            targets_indices = targets.squeeze().view(-1)
        # print(targets_indices,pred_score)
        score = 1.0
        try:
            score = roc_auc_score(targets_indices.cpu(), pred_score.cpu())
            return score
        except ValueError:
            pass
        return score


class PRAUC(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets_indices = torch.argmax(targets, dim=1).cpu()
            mask = targets == 1
            pred_score = preds[mask]
            pred_score = pred_score.view(-1).cpu()
        else:
            pred_score = preds.squeeze().view(-1).cpu()
            targets_indices = targets.squeeze().view(-1).cpu()
        # print(targets_indices,pred_score)
        score = 1.0
        try:
            precision, recall, threshold = precision_recall_curve(
                y_true=targets_indices, probas_pred=pred_score
            )
            score = auc(recall, precision)
            return score
        except ValueError:
            pass
        return score


class PearsonR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        try:
            shifted_x = preds - torch.mean(preds, dim=0)
            shifted_y = targets - torch.mean(targets, dim=0)
            sigma_x = torch.sqrt(torch.sum(shifted_x**2, dim=0))
            sigma_y = torch.sqrt(torch.sum(shifted_y**2, dim=0))

            pearson = torch.sum(shifted_x * shifted_y, dim=0) / (
                sigma_x * sigma_y + 1e-8
            )
            pearson = torch.clamp(pearson, min=-1, max=1)
            pearson = pearson.mean()
            return pearson
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, handle the error in a specific way,
            # like returning a default score or re-raising the exception.
            # For now, let's return a default score of 0.
            return 0


class MAE(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        loss = F.l1_loss(preds, targets)
        return loss


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        mse = torch.mean((preds - targets) ** 2)
        rmse = torch.sqrt(mse)
        return rmse


class ClassificationAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_only = True

    def forward(self, preds, targets):
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        try:
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                targets_indices = torch.argmax(targets, dim=1).cpu()
                pred_classes = torch.argmax(preds, dim=1).cpu()
            else:
                pred_classes = np.around(preds.squeeze().view(-1).cpu())
                targets_indices = targets.squeeze().view(-1).cpu()
            return accuracy_score(y_true=targets_indices, y_pred=pred_classes)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(preds, targets)
            # Optionally, handle the error in a specific way,
            # like returning a default score or re-raising the exception.
            # For now, let's return a default score of 0.
            return 0


class SpearmanR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        #         print(preds,targets)
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        preds_rank = torch.argsort(torch.argsort(preds, dim=0), dim=0).float()
        targets_rank = torch.argsort(torch.argsort(targets, dim=0), dim=0).float()

        shifted_x = preds_rank - torch.mean(preds_rank, dim=0)
        shifted_y = targets_rank - torch.mean(targets_rank, dim=0)
        sigma_x = torch.sqrt(torch.sum(shifted_x**2, dim=0))
        sigma_y = torch.sqrt(torch.sum(shifted_y**2, dim=0))

        spearman = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + 1e-8)
        spearman = torch.clamp(spearman, min=-1, max=1)
        spearman = spearman.mean()
        return spearman


class SpearmanCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = preds.detach().numpy().reshape(-1)
        targets = targets.detach().numpy().reshape(-1)
        preds = torch.squeeze(preds)
        targets = torch.squeeze(targets)
        return stats.spearmanr(preds, targets)[0]


def denormalize(normalized: torch.tensor, means, stds, eV2meV):
    denormalized = normalized * stds[None, :] + means[None, :]  # [batchsize, n_tasks]
    if eV2meV:
        denormalized = denormalized * eV2meV[None, :]
    return denormalized
