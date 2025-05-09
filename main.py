# main.py (Refactored with Pydantic)

import argparse
import datetime
import json
import os
import sys  # To combine args
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict, load_checkpoint

import register_model  # Ensure this imports build_model to register models
from config import TrainingConfig
from dataset_loader import RASampler, build_dataset, new_data_aug_generator
from src.evaluate import evaluate
from src.train import train_one_epoch
from src.utilities import utils
from src.utilities.utils import DistillationLoss


def get_runtime_args_parser():
    parser = argparse.ArgumentParser("SHViT runtime script", add_help=False)
    parser.add_argument(
        "--model", required=True, type=str, help="Name of model to train/eval"
    )
    parser.add_argument("--data-path", required=True, type=str, help="Dataset path")

    parser.add_argument("--resume", default="", help="Resume from checkpoint file path")
    parser.add_argument(
        "--input-size",
        type=int,
        help="Override default input size (usually from model config)",
    )
    parser.add_argument(
        "--weight-decay", type=float, help="Override default weight decay"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist-eval", action="store_true", help="Enable distributed evaluation"
    )
    return parser


def main():
    config = TrainingConfig()

    parser = argparse.ArgumentParser(
        "SHViT Training and Evaluation", parents=[get_runtime_args_parser()]
    )
    runtime_args = parser.parse_args()

    combined_args_dict = config.dict(by_alias=True)
    runtime_args_dict = vars(runtime_args)

    for key, value in runtime_args_dict.items():
        if value is not None:
            combined_args_dict[key] = value

    args = argparse.Namespace(**combined_args_dict)

    utils.init_distributed_mode(args)
    print("Effective configuration:")
    print(json.dumps(args.__dict__, indent=2))

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.repeated_aug:
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.three_augment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        distillation=(args.distillation_type != "none"),
        pretrained=False,
        fuse=False,
    )

    load_path = None
    if args.finetune:
        load_path = args.finetune
        print(f"Finetuning from: {load_path}")
    elif args.resume and not args.eval:
        load_path = args.resume
        print(f"Resuming from: {load_path}")
    elif args.resume and args.eval:
        load_path = args.resume
        print(f"Loading for eval from: {load_path}")

    if load_path:
        load_checkpoint(model, load_path, use_ema=args.model_ema)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using Model EMA.")

    model_without_ddp = model
    if args.distributed:  # Use args.distributed from utils
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    if args.lr is not None:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
        print(f"Scaled learning rate: {args.lr}")

    optimizer = create_optimizer(args, model_without_ddp)  # Pass combined args
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)  # Pass combined args

    criterion = LabelSmoothingCrossEntropy()
    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:  # Ensure float comparison
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != "none":
        assert args.teacher_path, "need to specify teacher-path when using distillation"
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=True,  # Assume teacher is pretrained from timm/torchhub
            num_classes=args.nb_classes,
            global_pool="avg",
        )
        if args.teacher_path and args.teacher_path.lower() != "none":
            if os.path.exists(args.teacher_path):
                print(f"Loading teacher weights from local path: {args.teacher_path}")
                load_checkpoint(teacher_model, args.teacher_path)
            elif args.teacher_path.startswith("http"):
                print(f"Loading teacher weights from URL: {args.teacher_path}")
                load_checkpoint(
                    teacher_model, args.teacher_path, use_ema=False
                )  # Assuming URL doesn't point to EMA usually
            else:
                print(
                    f"Warning: teacher_path {args.teacher_path} not found or not URL, using default pretrained {args.teacher_model}"
                )

        teacher_model.to(device)
        teacher_model.eval()

    criterion = DistillationLoss(
        criterion,
        teacher_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config_effective.txt").open("w") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
    resume_epoch = None
    if args.resume and not args.eval and not args.finetune:
        resume_epoch = utils.resume_checkpoint(
            model_without_ddp,
            args.resume,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=utils.is_main_process(),
        )
        # Resume EMA if applicable
        if model_ema is not None:
            utils.resume_checkpoint(
                model_ema.module, args.resume, use_ema=True, log_info=False
            )  # Load EMA state

        if resume_epoch is not None:
            print(f"Resumed optimizer/scaler state from epoch {resume_epoch}")
            args.start_epoch = resume_epoch + 1  # Override start epoch
            lr_scheduler.step(args.start_epoch)

    if args.eval:
        if load_path is None:
            print(
                "Warning: Evaluation requested but no checkpoint specified via --resume."
            )
            return
        print(f"Evaluating model from: {load_path}")
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    print(f"Start training from epoch {args.start_epoch} for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            args.clip_mode,
            model_ema,
            mixup_fn,
            set_training_mode=args.finetune == "",  # Set to False only if finetuning
            set_bn_eval=args.set_bn_eval,
        )

        lr_scheduler.step(epoch + 1)  # Step scheduler after epoch

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy epoch {epoch}: Acc@1 {test_stats['acc1']:.3f}")

        if output_dir:
            checkpoint_paths = [output_dir / "checkpoint_last.pth"]
            if test_stats["acc1"] > max_accuracy:
                max_accuracy = test_stats["acc1"]
                checkpoint_paths.append(output_dir / "checkpoint_best.pth")
                print(f"*** New Best Acc@1: {max_accuracy:.3f} at epoch {epoch} ***")

            if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
                checkpoint_paths.append(output_dir / f"checkpoint_{epoch:04d}.pth")

            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,  # Save combined args
                }
                if model_ema is not None:
                    save_dict["model_ema"] = get_state_dict(model_ema)
                if loss_scaler is not None:
                    save_dict["scaler"] = loss_scaler.state_dict()

                utils.save_on_master(save_dict, checkpoint_path)
        else:
            max_accuracy = max(max_accuracy, test_stats["acc1"])

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(f"Max accuracy: {max_accuracy:.2f}%")


if __name__ == "__main__":
    main()
