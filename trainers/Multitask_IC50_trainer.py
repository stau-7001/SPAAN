import os
from itertools import chain
from typing import Dict, Callable
import torch
from models import *  # do not remove
from torch.utils.data import DataLoader
from trainers.trainer import Trainer
from typing import Tuple, Dict, Callable, Union
from commons.utils import (
    flatten_dict,
    tensorboard_gradient_magnitude,
    move_to_device,
    update_matrix,
)
import json


class MultitaskIC50Trainer(Trainer):
    def __init__(
        self,
        model,
        args,
        metrics: Dict[str, Callable],
        main_metric: str,
        device: torch.device,
        tensorboard_functions: Dict[str, Callable],
        optim=None,
        main_metric_goal: str = "min",
        loss_func=torch.nn.MSELoss,
        scheduler_step_per_batch: bool = True,
        **kwargs,
    ):
        super(MultitaskIC50Trainer, self).__init__(
            model,
            args,
            metrics,
            main_metric,
            device,
            tensorboard_functions,
            optim,
            main_metric_goal,
            loss_func,
            scheduler_step_per_batch,
        )

    def forward_pass(self, batch, precompute=True):
        #         print(batch[0])
        if precompute:
            reg_pred, cls_pred, combined_representation = self.model(batch)
        else:
            reg_pred, cls_pred, combined_representation = self.model(
                batch["H"], batch["L"], batch["spike"]
            )  # forward the sequence to the model
        reg_label = torch.FloatTensor(batch["IC50"]).to(self.device)
        cls_label = torch.FloatTensor(batch["cls_label"]).to(self.device)
        #         print(pred.shape, label.shape, label.float().unsqueeze(0).shape)
        loss = self.loss_func(
            reg_pred, cls_pred, reg_label.float().unsqueeze(0), cls_label
        )
        # print(reg_pred, cls_pred, reg_label,cls_label)
        return (
            loss,
            reg_pred,
            cls_pred,
            reg_label.float().unsqueeze(0),
            cls_label,
            combined_representation,
        )

    def process_batch(self, batch, optim):
        loss, reg_pred, cls_pred, reg_label, cls_label, _ = self.forward_pass(batch)
        # cls_label = torch.tensor(cls_label, dtype=torch.float).squeeze(0)
        if optim != None:  # run backpropagation if an optimizer is provided (ie.train)
            loss.backward()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return (
            loss,
            reg_pred.detach(),
            cls_pred.detach(),
            reg_label.detach(),
            cls_label.detach(),
        )

    def predict(
        self,
        data_loader: DataLoader,
        epoch: int,
        optim: torch.optim.Optimizer = None,
        return_predictions: bool = False,
    ) -> Union[
        Dict, Tuple[float, Union[torch.Tensor, None], Union[torch.Tensor, None]]
    ]:
        total_metrics = {
            k: 0
            for k in list(self.metrics.keys())
            + [
                type(self.loss_func).__name__,
                "mean_pred",
                "std_pred",
                "mean_targets",
                "std_targets",
            ]
        }
        reg_epoch_targets = torch.tensor([]).to(self.device)
        reg_epoch_predictions = torch.tensor([]).to(self.device)
        cls_epoch_targets = torch.tensor([]).to(self.device)
        cls_epoch_predictions = torch.tensor([]).to(self.device)
        reg_accumulated_preds = torch.tensor([]).to(self.device)
        reg_accumulated_targets = torch.tensor([]).to(self.device)
        cls_accumulated_preds = torch.tensor([]).to(self.device)
        cls_accumulated_targets = torch.tensor([]).to(self.device)
        epoch_loss = 0

        for i, batch in enumerate(data_loader):
            batch = move_to_device(batch, self.device)

            loss, reg_predictions, cls_predictions, reg_targets, cls_targets = (
                self.process_batch(batch, optim)
            )
            with torch.no_grad():

                reg_accumulated_preds = torch.cat(
                    (reg_predictions, reg_accumulated_preds), 0
                )
                reg_accumulated_targets = torch.cat(
                    (reg_targets, reg_accumulated_targets), 0
                )
                cls_accumulated_preds = torch.cat(
                    (cls_predictions, cls_accumulated_preds), 0
                )
                cls_accumulated_targets = torch.cat(
                    (cls_targets, cls_accumulated_targets), 0
                )

                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    if self.optim_steps % self.accumulation_steps == 0:  # predict
                        metrics = self.evaluate_metrics(
                            reg_accumulated_preds,
                            cls_accumulated_preds,
                            reg_accumulated_targets,
                            cls_accumulated_targets,
                        )
                        metrics[type(self.loss_func).__name__] = loss.item()

                        self.run_tensorboard_functions(
                            reg_accumulated_preds,
                            reg_accumulated_targets,
                            step=self.optim_steps,
                            data_split="train",
                        )
                        self.tensorboard_log(
                            metrics,
                            data_split="train",
                            step=self.optim_steps,
                            epoch=epoch,
                        )
                        reg_accumulated_preds = torch.tensor([]).to(self.device)
                        reg_accumulated_targets = torch.tensor([]).to(self.device)
                        cls_accumulated_preds = torch.tensor([]).to(self.device)
                        cls_accumulated_targets = torch.tensor([]).to(self.device)

                        metrics_str = ", ".join(
                            [f"{key}: {value:.7f}" for key, value in metrics.items()]
                        )
                        print(
                            f"[Epoch {epoch}; Iter {i + 1:5d}/{len(data_loader):5d}] train: {metrics_str}"
                        )
                if (
                    optim == None and self.val_per_batch
                ):  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = self.evaluate_metrics(
                        reg_accumulated_preds,
                        cls_accumulated_preds,
                        reg_accumulated_targets,
                        cls_accumulated_targets,
                        val=True,
                    )
                    metrics_results[type(self.loss_func).__name__] = loss.item()
                    if i == 0 and epoch in self.args.models_to_save:
                        self.run_tensorboard_functions(
                            accumulated_preds,
                            accumulated_targets,
                            step=self.optim_steps,
                            data_split="val",
                        )
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    epoch_loss += loss.item()
                    reg_epoch_targets = torch.cat((reg_targets, reg_epoch_targets), 0)
                    reg_epoch_predictions = torch.cat(
                        (reg_predictions, reg_epoch_predictions), 0
                    )
                    cls_epoch_targets = torch.cat((cls_targets, cls_epoch_targets), 0)
                    cls_epoch_predictions = torch.cat(
                        (cls_predictions, cls_epoch_predictions), 0
                    )

        if optim == None:
            if self.val_per_batch:
                total_metrics = {
                    k: v / len(data_loader) for k, v in total_metrics.items()
                }
            else:
                total_metrics = self.evaluate_metrics(
                    reg_epoch_predictions,
                    cls_epoch_predictions,
                    reg_epoch_targets,
                    cls_epoch_targets,
                    val=True,
                )
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(
                    data_loader
                )
            return total_metrics

    def initialize_optimizer(self, optim):
        # print("model", self.model.named_parameters())
        normal_params = [v for k, v in chain(self.model.named_parameters())]
        self.optim = optim([{"params": normal_params}], **self.args.optimizer_params)

    def evaluate_metrics(
        self,
        reg_predictions,
        cls_predictions,
        reg_targets,
        cls_targets,
        batch=None,
        val=False,
    ) -> Dict[str, float]:
        metrics = {}
        import csv

        # with open('predictions.csv', 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     for prediction in reg_predictions:
        #         csvwriter.writerow([prediction.item()])
        metrics[f"mean_pred"] = torch.mean(reg_predictions).item()
        metrics[f"std_pred"] = torch.std(reg_predictions).item()
        metrics[f"mean_targets"] = torch.mean(reg_targets).item()
        metrics[f"std_targets"] = torch.std(reg_targets).item()
        for key, metric in self.metrics.items():
            if hasattr(metric, "cls_only"):
                metrics[key] = update_matrix(metric(cls_predictions, cls_targets))
            elif not hasattr(metric, "val_only") or val:
                metrics[key] = update_matrix(metric(reg_predictions, reg_targets))

        return metrics

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save(
            {
                "epoch": epoch,
                "best_val_score": self.best_val_score,
                "optim_steps": self.optim_steps,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "scheduler_state_dict": (
                    None
                    if self.lr_scheduler == None
                    else self.lr_scheduler.state_dict()
                ),
            },
            os.path.join(self.writer.log_dir, checkpoint_name),
        )
