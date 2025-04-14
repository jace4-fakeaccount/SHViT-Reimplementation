"""
Train and eval functions used in main.py
"""

import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

from utilities import utils


def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: utils.DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    clip_grad: float = 0,
    clip_mode: str = "norm",
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    set_bn_eval=False,
):
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if True:  # with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=clip_grad,
            clip_mode=clip_mode,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
