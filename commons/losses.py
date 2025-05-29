import itertools
import math
import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss, MSELoss
import numpy as np
import torch.nn.functional as F


class CpLoss(nn.Module):
    def __init__(
        self,
        initial_temperature,
        annealing_epochs,
        entropy_weight=1.0,
        loss_weight=5.0,
        q=2,
        class_num=11,
    ):
        super(CpLoss, self).__init__()
        # self.smax = smax  # Softmax function or method
        self.initial_temperature = initial_temperature
        self.annealing_epochs = annealing_epochs
        self.entropy_weight = entropy_weight
        self.loss_weight = loss_weight
        self.q = q
        # self.range_vals = (
        #     range_vals  # Reference values for the categories (e.g., quantiles)
        # )
        interval_len = 10 / (class_num - 1)
        self.range_vals = torch.arange(
            interval_len * 0.5, 10 + interval_len * 1.5, interval_len
        )  # Reference values for the categories (e.g., quantiles)
        self.annealing = False  # Set to True to enable annealing mechanism
        # self.current_epoch = 0  # This should be updated during training

    def forward(self, probs, targets, current_epoch):
        # probs += 1e-8
        targets = torch.where(targets > 10.0, torch.tensor(10.0), targets)
        # Apply entropy annealing
        if self.annealing:
            self.entropy_weight = max(
                0,
                (
                    self.initial_temperature
                    * (1 - current_epoch / self.annealing_epochs)
                ),
            )

        # Entropy regularization term
        probs = torch.clamp(probs, 1e-10, 1.0 - 1e-10)
        log_vals = torch.log(probs)
        log_vals[probs == 0] = 0  # To avoid log(0)
        neg_entropy = torch.sum(torch.sum(probs * log_vals, dim=1))
        all_losses = [neg_entropy * self.entropy_weight]

        # Expectation error term
        all_losses.append(
            self.loss_weight
            * torch.sum(
                probs
                * torch.pow(
                    torch.abs(
                        self.range_vals.view(1, -1)
                        .expand(len(targets), -1)
                        .to(targets.device)
                        - targets.view(-1, 1)
                    ),
                    self.q,
                )
            )
        )

        # Final loss calculation
        loss = torch.sum(torch.stack(all_losses))
        return loss


class CustomL1Loss(_Loss):
    def __init__(self, w1=2.0, w2=0.8, w3=2.0, w4=0.01) -> None:
        super(CustomL1Loss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def forward(self, preds, targets):
        mask1 = (targets < 10) & (preds < 10)
        mask2 = (targets >= 10) & (preds < 10)
        mask3 = (targets < 10) & (preds >= 10)
        mask4 = (targets >= 10) & (preds >= 10)

        loss1 = F.l1_loss(preds[mask1], targets[mask1], reduction="none")
        loss2 = F.l1_loss(preds[mask2], targets[mask2], reduction="none")
        loss3 = F.l1_loss(preds[mask3], targets[mask3], reduction="none")
        loss4 = F.l1_loss(preds[mask4], targets[mask4], reduction="none")

        loss = (
            self.w1 * loss1.sum()
            + self.w2 * loss2.sum()
            + self.w3 * loss3.sum()
            + self.w4 * loss4.sum()
        )
        loss = loss / (
            mask1.float().sum()
            + mask2.float().sum()
            + mask3.float().sum()
            + mask4.float().sum()
        )

        return loss


class CustomMSELoss(nn.Module):
    def __init__(self, w1=2.0, w2=0.8, w3=2.0, w4=0.01):
        super(CustomMSELoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def forward(self, preds, targets):
        mask1 = (targets < 10) & (preds < 10)
        mask2 = (targets >= 10) & (preds < 10)
        mask3 = (targets < 10) & (preds >= 10)
        mask4 = (targets >= 10) & (preds >= 10)

        loss1 = F.mse_loss(preds[mask1], targets[mask1], reduction="none")
        loss2 = F.mse_loss(preds[mask2], targets[mask2], reduction="none")
        loss3 = F.mse_loss(preds[mask3], targets[mask3], reduction="none")
        loss4 = F.mse_loss(preds[mask4], targets[mask4], reduction="none")

        loss = (
            self.w1 * loss1.sum()
            + self.w2 * loss2.sum()
            + self.w3 * loss3.sum()
            + self.w4 * loss4.sum()
        )
        loss = loss / (
            mask1.float().sum()
            + mask2.float().sum()
            + mask3.float().sum()
            + mask4.float().sum()
        )

        return loss


class RegFocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(RegFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1):
        pred = pred.reshape(1, -1)
        target = target.reshape(1, -1)
        se_ = torch.abs(pred - target)
        a_ = torch.pow(se_, gamma)
        a_sum = torch.sum(a_) + 1e-6
        a_ = a_ / a_sum
        if torch.isnan(se_).any() or torch.isinf(se_).any():
            print("NaN or Inf detected in se_")
        if torch.isnan(a_).any() or torch.isinf(a_).any():
            print("NaN or Inf detected in a_")

        return torch.sum(torch.pow(se_, 2) * a_) / len(pred + 1e-6)


class ClsFocalLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(ClsFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1, alpha=0.5):
        assert alpha < 1 and alpha > 0

        epsilon = 1e-7
        pred = pred.reshape(
            -1,
        )
        target = target.reshape(
            -1,
        )

        pt_0 = 1 - pred[target == 0]
        pt_1 = pred[target == 1]

        loss_0 = (-torch.pow(1 - pt_0, gamma) * torch.log(pt_0 + epsilon)).sum()
        loss_1 = (-torch.pow(1 - pt_1, gamma) * torch.log(pt_1 + epsilon)).sum()

        loss = (1 - alpha) * loss_0 + alpha * loss_1

        return loss / len(pred)


class Reg_Cls_Loss01(_Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        gamma: int = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        super(Reg_Cls_Loss01, self).__init__(size_average, reduce, reduction)
        assert alpha < 1 and alpha > 0
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.Cls_loss = ClsFocalLoss(size_average, reduce, reduction)
        self.Reg_loss = RegFocalLoss(size_average, reduce, reduction)

    def forward(self, reg_pred, cls_pred, reg_target, cls_target):
        cls_loss = self.Cls_loss(cls_pred, cls_target, self.gamma, self.alpha)
        reg_loss = self.Reg_loss(reg_pred, reg_target, self.gamma)
        return self.beta * reg_loss + (1 - self.beta) * cls_loss


class Reg_Cls_Loss(_Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        gamma: int = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        super(Reg_Cls_Loss, self).__init__(size_average, reduce, reduction)
        assert alpha < 1 and alpha > 0
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, reg_pred, cls_pred, reg_target, cls_target):
        epsilon = 1e-7
        cls_pred = cls_pred.reshape(
            -1,
        )
        cls_target = cls_target.reshape(
            -1,
        )

        #         print(cls_pred,cls_target)
        pt_0 = 1 - cls_pred[cls_target == 0]
        pt_1 = cls_pred[cls_target == 1]

        loss_0 = (-torch.pow(1 - pt_0, self.gamma) * torch.log(pt_0 + epsilon)).sum()
        loss_1 = (-torch.pow(1 - pt_1, self.gamma) * torch.log(pt_1 + epsilon)).sum()
        cls_loss = (1 - self.alpha) * loss_0 + self.alpha * loss_1

        reg_pred = reg_pred.reshape(1, -1)
        reg_target = reg_target.reshape(1, -1)
        se_ = torch.abs(reg_pred - reg_target) + epsilon
        a_ = torch.pow(se_, self.gamma)
        a_sum = torch.sum(a_) + epsilon
        a_ = a_ / a_sum
        reg_loss = torch.sum(torch.pow(se_, 2) * a_)

        return (self.beta * reg_loss + (1 - self.beta) * cls_loss) / len(reg_pred)
