import math
import random
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import labram_utils as utils
from eeg_data.dataset import EEGDataset
from utils.misc import WandbLogger

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, dataset: EEGDataset,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, metric_logger: utils.MetricLogger = None,
                    log_writer: WandbLogger = None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    alpha = 0.8
    correct, total = 0, 0
    img_features_all = dataset.img_features
    img_features_all = img_features_all.to(device, non_blocking=True)

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (inp_eeg, _, labels, _, text_features, img_features) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = inp_eeg.float().to(device, non_blocking=True)
        img_features = img_features.to(device, non_blocking=True)
        text_features = text_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type='cuda'):
            eeg_features, img_loss, text_loss = model(samples, input_chans=input_chans, img_features=img_features, text_features=text_features)
            regress_loss =  torch.nn.MSELoss()(eeg_features, img_features)
            loss = (alpha * regress_loss *10 + (1 - alpha) * img_loss*10)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        logit_scale = model.module.logit_scale
        logits_img = logit_scale * eeg_features.float() @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) ∈ {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        acc = correct / total
        
        torch.cuda.synchronize()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(acc=acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(train_loss=loss_value, head="metrics")
            log_writer.update(train_loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(dataset: EEGDataset, data_loader, model, device, header='Test:', ch_names=None, 
             metric_logger: utils.MetricLogger = None, log_writer: WandbLogger = None):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if metric_logger is None:
        metric_logger = utils.MetricLogger(delimiter="  ")
    alpha = 0.8
    #header = 'Test:'

    text_features_all = dataset.text_features
    img_features_all = dataset.img_features
    img_features_all = img_features_all.to(device, non_blocking=True)
    all_labels = set(range(text_features_all.size(0)))
    correct = 0
    total = 0
    top5_correct = 0
    top5_correct_count = 0
    # switch to evaluation mode
    model.eval()
    
    for step, (inp_eeg, _, labels, _, text_features, img_features) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        inp_eeg = inp_eeg.to(device, non_blocking=True).float()
        img_features = img_features.to(device, non_blocking=True)
        text_features = text_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(device_type='cuda'):
            eeg_features, img_loss, text_loss = model(inp_eeg, input_chans=input_chans, img_features=img_features, text_features=text_features)
            regress_loss =  torch.nn.MSELoss()(eeg_features, img_features)
            loss = (alpha * regress_loss *10 + (1 - alpha) * img_loss*10)
            loss_value = loss.item()

        for idx, label in enumerate(labels):
            k = 200
            # First select k-1 classes excluding the correct class
            possible_classes = list(all_labels - {label.item()})
            selected_classes = random.sample(possible_classes, k-1) + [label.item()]
            selected_img_features = img_features_all[selected_classes]
            selected_text_features = text_features_all[selected_classes]
            
            # Compute corresponding logits
            logit_scale = model.module.logit_scale
            logits_img = logit_scale * eeg_features[idx].float() @ selected_img_features.T
            logits_single = logits_img
            # print("logits_single", logits_single.shape)
            # Get predicted class
            # predicted_label = selected_classes[torch.argmax(logits_single).item()]
            predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
            if predicted_label == label.item():
                # print("predicted_label", predicted_label)
                correct += 1
            
            # logits_single is the model output, assumed to be shape (n_batch, n_classes)
            # label is the true label, shape (n_batch,)
            # Get top-5 predicted indices
            # print("logits_single", logits_single)
            _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                    
            # Check if true label is in top-5 predictions
            if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                top5_correct_count+=1                                
            total += 1
    accuracy = correct / total
    top5_acc = top5_correct_count / total
        
    metric_logger.update(val_loss=loss_value)
    metric_logger.update(val_acc=accuracy)
    metric_logger.update(val_top5_acc=top5_acc)
    print(f"accuracy: {accuracy}, top5_acc: {top5_acc}")
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if log_writer is not None:
        log_writer.update(val_loss=metric_logger.val_loss.value, head="metrics", step=log_writer.step)
        log_writer.update(val_acc=metric_logger.val_acc.value, head="metrics", step=log_writer.step)
        log_writer.update(val_top5_acc=metric_logger.val_top5_acc.value, head="metrics", step=log_writer.step)
        log_writer.set_step()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}