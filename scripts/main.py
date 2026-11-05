import argparse
import yaml
from dataclasses import dataclass
from typing import Optional

from utils.model_utils import setup_seed, run_finetuning, run_pretraining, run_diffusion

@dataclass
class MetaConfig:
    task: str
    data_mode: str
    seed: int
    cuda: int

@dataclass
class ArchConfig:
    in_dim: int
    num_classes: int
    d_model: int
    dim_feedforward: int
    seq_len: int
    n_layer: int
    nhead: int

@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    clip_value: float
    lr_scheduler: str
    pretrained_weights: Optional[str] = None
    diffusion_weights: Optional[str] = None
    synthetic_ratio: int = 1
    cfg_scale: float = 4.0

@dataclass
class PathConfig:
    model_dir: str
    datasets_dir: str
    synthetic_data_dir: str

@dataclass
class PipelineConfig:
    meta: MetaConfig
    arch: ArchConfig
    training: TrainConfig
    paths: PathConfig
    
    def get_flat_params(self):
        class FlatParams: pass
        p = FlatParams()
        for cfg in [self.meta, self.arch, self.training, self.paths]:
            for k, v in cfg.__dict__.items():
                setattr(p, k, v)
        p.parallel = False
        p.need_mask = False
        return p

def main():
    parser = argparse.ArgumentParser(description='Universal Pipeline Controller')
    parser.add_argument('--config', type=str, default='config.yml')
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as file:
        raw_yaml = yaml.safe_load(file)

    config = PipelineConfig(
        meta=MetaConfig(**raw_yaml.get('meta', {})),
        arch=ArchConfig(**raw_yaml.get('arch', {})),
        training=TrainConfig(**raw_yaml.get('training', {})),
        paths=PathConfig(**raw_yaml.get('paths', {}))
    )

    setup_seed(config.meta.seed)
    print(f"--- Starting Task: {config.meta.task.upper()} ---")

    if config.meta.task == 'finetune':
        run_finetuning(config)
    elif config.meta.task == 'pretrain':
        run_pretraining(config)
    elif config.meta.task == 'train_diffusion':
        run_diffusion(config)
    else:
        print(f"Task '{config.meta.task}' not recognized.")

if __name__ == '__main__':
    main()