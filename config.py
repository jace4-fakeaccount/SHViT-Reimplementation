from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, DirectoryPath, Field, FilePath


class ClipMode(str, Enum):
    norm = "norm"
    value = "value"
    agc = "agc"


class SchedulerType(str, Enum):
    cosine = "cosine"
    step = "step"
    plateau = "plateau"


class OptimizerType(str, Enum):
    adamw = "adamw"
    sgd = "sgd"


class InterpolationType(str, Enum):
    random = "random"
    bilinear = "bilinear"
    bicubic = "bicubic"


class RemodeType(str, Enum):
    pixel = "pixel"
    const = "const"


class MixupMode(str, Enum):
    batch = "batch"
    pair = "pair"
    elem = "elem"


class DistillationType(str, Enum):
    none = "none"
    soft = "soft"
    hard = "hard"


class DatasetType(str, Enum):
    cifar = "CIFAR"
    imnet = "IMNET"
    inat = "INAT"
    inat19 = "INAT19"


class INatCategory(str, Enum):
    kingdom = "kingdom"
    phylum = "phylum"
    class_ = "class"
    order = "order"
    supercategory = "supercategory"
    family = "family"
    genus = "genus"
    name = "name"


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=256, alias="batch-size")
    epochs: int = 300
    output_dir: Optional[str] = ""
    device: str = "cuda"
    seed: int = 0
    num_workers: int = 10
    pin_mem: bool = Field(default=True, alias="pin-mem")
    start_epoch: int = Field(default=0, alias="start_epoch")
    save_freq: int = Field(default=1, alias="save_freq")

    # Model EMA
    model_ema: bool = Field(default=True, alias="model-ema")
    model_ema_decay: float = Field(default=0.99996, alias="model-ema-decay")
    model_ema_force_cpu: bool = Field(default=False, alias="model-ema-force-cpu")

    # Optimizer
    opt: OptimizerType = OptimizerType.adamw
    opt_eps: Optional[float] = Field(default=1e-8, alias="opt-eps")
    opt_betas: Optional[Tuple[float, float]] = Field(default=None, alias="opt-betas")
    momentum: float = 0.9  # Only used for SGD-like optimizers
    clip_grad: Optional[float] = Field(default=0.02, alias="clip-grad")
    clip_mode: ClipMode = Field(default=ClipMode.agc, alias="clip-mode")

    # Scheduler
    sched: SchedulerType = SchedulerType.cosine
    lr: float = 1e-3
    lr_noise: Optional[List[float]] = Field(default=None, alias="lr-noise")
    lr_noise_pct: float = Field(default=0.67, alias="lr-noise-pct")
    lr_noise_std: float = Field(default=1.0, alias="lr-noise-std")
    warmup_lr: float = Field(default=1e-6, alias="warmup-lr")
    min_lr: float = Field(default=1e-5, alias="min-lr")
    decay_epochs: float = Field(default=30, alias="decay-epochs")
    warmup_epochs: int = Field(default=5, alias="warmup-epochs")
    cooldown_epochs: int = Field(default=10, alias="cooldown-epochs")
    patience_epochs: int = Field(default=10, alias="patience-epochs")
    decay_rate: float = Field(default=0.1, alias="decay-rate")

    # Augmentation & Regularization
    three_augment: bool = Field(default=False, alias="ThreeAugment")
    color_jitter: Optional[float] = Field(default=0.4, alias="color-jitter")
    aa: Optional[str] = Field(
        default="rand-m9-mstd0.5-inc1", alias="aa"
    )  # AutoAugment policy as string
    smoothing: float = 0.1
    train_interpolation: InterpolationType = Field(
        default=InterpolationType.bicubic, alias="train-interpolation"
    )
    repeated_aug: bool = Field(default=True, alias="repeated-aug")
    reprob: float = Field(default=0.25)  # Random erase prob
    remode: RemodeType = Field(default=RemodeType.pixel)  # Random erase mode
    recount: int = Field(default=1)  # Random erase count
    resplit: bool = Field(default=False)  # Random erase split

    # Mixup
    mixup: float = 0.8
    cutmix: float = 1.0
    cutmix_minmax: Optional[List[float]] = Field(default=None, alias="cutmix-minmax")
    mixup_prob: float = Field(default=1.0, alias="mixup-prob")
    mixup_switch_prob: float = Field(default=0.5, alias="mixup-switch-prob")
    mixup_mode: MixupMode = Field(default=MixupMode.batch, alias="mixup-mode")

    # Distillation
    teacher_model: Optional[str] = Field(default="regnety_160", alias="teacher-model")
    teacher_path: Optional[str] = Field(
        default="https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth",
        alias="teacher-path",
    )
    distillation_type: DistillationType = Field(
        default=DistillationType.none, alias="distillation-type"
    )
    distillation_alpha: float = Field(default=0.5, alias="distillation-alpha")
    distillation_tau: float = Field(default=1.0, alias="distillation-tau")

    # Finetuning
    finetune: str = ""  # Keep as string for path/URL
    set_bn_eval: bool = Field(default=False, alias="set_bn_eval")

    # Dataset (specific path/name are handled by argparse)
    data_set: DatasetType = Field(default=DatasetType.imnet, alias="data-set")
    inat_category: Optional[INatCategory] = Field(
        default=INatCategory.name, alias="inat-category"
    )

    dist_url: str = Field(default="env://", alias="dist_url")

    class Config:
        use_enum_values = True
