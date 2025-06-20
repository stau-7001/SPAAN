import os
from itertools import chain
from typing import Dict, Callable
import torch
from models import *  # do not remove
from torch.utils.data import DataLoader
from trainers.trainer import Trainer
from typing import Tuple, Dict, Callable, Union
from commons.utils import flatten_dict, tensorboard_gradient_magnitude, move_to_device
import json

def update_matrix(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value 
    
class ClsTrainer(Trainer):
    def __init__(self, model, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        super(ClsTrainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                                               optim, main_metric_goal, loss_func, scheduler_step_per_batch)
        
    def forward_pass(self, batch):
#         print(batch[0])
        # cls_pred = self.model(batch['H'],batch['L'],batch['spike'])  # forward the sequence to the model
        # print(batch)
        cls_pred = self.model(batch)
        cls_label = torch.FloatTensor(batch['cls_label']).to(self.device)
#         print(pred.shape, label.shape, label.float().unsqueeze(0).shape)
        loss = self.loss_func(cls_pred,cls_label)
        # print(reg_pred, cls_pred, reg_label,cls_label)
        return loss, cls_pred ,cls_label

    def run_per_epoch_evaluations(self, data_loader):
        print('fitting linear probe')
        representations = []
        targets = []
        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, pred, label = self.process_batch(batch, optim=None)
            representations.append(pred)
            targets.append(label)
            if len(representations) * len(pred) >= self.args.linear_probing_samples:
                break
        representations = torch.cat(representations, dim=0)
        targets = torch.cat(targets, dim=0)
        if len(representations) >= representations.shape[-1]:
            X, _ = torch.lstsq(targets, representations)
            X, _ = torch.lstsq(targets, representations)
            sol = X[:representations.shape[-1]]
            pred = representations @ sol
            mean_absolute_error = (pred - targets).abs().mean()
            self.writer.add_scalar('linear_probe_mae', mean_absolute_error.item(), self.optim_steps)
        else:
            raise ValueError(
                f'We have less linear_probing_samples {len(representations)} than the metric dimension {representations.shape[-1]}. Linear probing cannot be used.')

        print('finish fitting linear probe')
    
    def process_batch(self, batch, optim):
        loss, cls_pred,cls_label = self.forward_pass(batch)
        # cls_label = torch.tensor(cls_label, dtype=torch.float).squeeze(0)
        if optim != None:  # run backpropagation if an optimizer is provided (ie.train)
            loss.backward()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return  loss, cls_pred.detach(), cls_label.detach()
    

    def predict(self, data_loader: DataLoader, epoch: int, optim: torch.optim.Optimizer = None,
        return_predictions: bool = False) -> Union[
        Dict, Tuple[float, Union[torch.Tensor, None], Union[torch.Tensor, None]]]:
        total_metrics = {k: 0 for k in
                         list(self.metrics.keys()) + [type(self.loss_func).__name__, 'mean_pred', 'std_pred',
                                                      'mean_targets', 'std_targets']}
        cls_epoch_targets = torch.tensor([]).to(self.device)
        cls_epoch_predictions = torch.tensor([]).to(self.device)
        cls_accumulated_preds = torch.tensor([]).to(self.device)
        cls_accumulated_targets = torch.tensor([]).to(self.device)
        epoch_loss = 0
#         dict_list = []

        for i, batch in enumerate(data_loader):
            #ic(self.optim.param_groups)
#             print(batch)
#             dicts = batch
            batch = move_to_device(batch, self.device)
    
            loss,cls_predictions, cls_targets = self.process_batch(batch, optim)
#             dicts = {**dicts, **cur_res}
# #             print(dicts)
#             dict_list.append(dicts)
            with torch.no_grad():
                cls_accumulated_preds = torch.cat((cls_predictions, cls_accumulated_preds), 0)
                cls_accumulated_targets = torch.cat((cls_targets, cls_accumulated_targets), 0)
                
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    if self.optim_steps % self.accumulation_steps == 0:# predict
                        metrics = self.evaluate_metrics(cls_accumulated_preds, cls_accumulated_targets)
                        metrics[type(self.loss_func).__name__] = loss.item()
                        
                        self.run_tensorboard_functions(cls_accumulated_preds, cls_accumulated_targets, step=self.optim_steps, data_split='train')
                        self.tensorboard_log(metrics, data_split='train', step=self.optim_steps, epoch=epoch)
                        cls_accumulated_preds = torch.tensor([]).to(self.device)
                        cls_accumulated_targets = torch.tensor([]).to(self.device)
                        
                        metrics_str = ', '.join([f'{key}: {value:.7f}' for key, value in metrics.items()])
                        print(f'[Epoch {epoch}; Iter {i + 1:5d}/{len(data_loader):5d}] train: {metrics_str}')
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = self.evaluate_metrics( cls_accumulated_preds,  cls_accumulated_targets, val=True)
                    metrics_results[type(self.loss_func).__name__] = loss.item()
                    if i ==0 and epoch in self.args.models_to_save:
                        self.run_tensorboard_functions(accumulated_preds, accumulated_targets, step=self.optim_steps, data_split='val')
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    epoch_loss += loss.item()
                    cls_epoch_targets = torch.cat((cls_targets, cls_epoch_targets), 0)
                    cls_epoch_predictions = torch.cat((cls_predictions, cls_epoch_predictions), 0)

        if optim == None:
            if self.val_per_batch:
                total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
            else:
                total_metrics = self.evaluate_metrics(cls_epoch_predictions, cls_epoch_targets, val=True)
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(data_loader)
#             with open("val_output.json", "w") as outfile:
#                 converted_dict_list = [{k: (v.cpu().tolist() if isinstance(v, torch.Tensor) else v) for k, v in d.items()} for d in dict_list]
#                 json.dump(converted_dict_list, outfile)
            return total_metrics
#         else:
#             with open("train_output.json", "w") as outfile:
#                 converted_dict_list = [{k: (v.cpu().tolist() if isinstance(v, torch.Tensor) else v) for k, v in d.items()} for d in dict_list]
#                 json.dump(converted_dict_list, outfile)


    def initialize_optimizer(self, optim):
        print(self.model.named_parameters())
        normal_params = [v for k, v in chain(self.model.named_parameters())]
#         batch_norm_params = [v for k, v in chain(self.model.named_parameters()) if
#                              'batch_norm' in k]
#         print(normal_params)
#         print(batch_norm_params)
        self.optim = optim([{'params': normal_params}],
                           **self.args.optimizer_params)
    
    def evaluate_metrics(self, cls_predictions, cls_targets, batch=None, val=False) -> Dict[str, float]:
        metrics = {}
        for key, metric in self.metrics.items():
            metrics[key] = update_matrix(metric(cls_predictions, cls_targets))

        return metrics
    
    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
