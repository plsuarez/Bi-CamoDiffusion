from utils import init_env

import argparse
import torch

from torch.utils.data import DataLoader

from utils.collate_utils import collate, SampleDataset
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from utils.trainer import Trainer

set_random_seed(42)


# -----------------------------
# helpers para cfg (dict/attr)
# -----------------------------
def _has(obj, key: str) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return key in obj
    return hasattr(obj, key)

def _get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _set(obj, key: str, value):
    if obj is None:
        return
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)

def _pop(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.pop(key, default)
    if hasattr(obj, key):
        val = getattr(obj, key)
        delattr(obj, key)
        return val
    return default


def get_loader(cfg):
    train_dataset = instantiate_from_config(cfg.train_dataset)

    # 
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    
    test_dataset = instantiate_from_config(cfg.test_dataset.CAMO)

    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.COD10K), interval=10)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])

    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.NC4K), interval=30)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default=None, help='None for saving in wandb folder.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_min', type=float, default=1e-6)

    cfg = add_args(parser)
    config_pretty(cfg)


    # ---------------------------------------------------------
    diff_params = _get(cfg.diffusion_model, "params", None)
    if diff_params is not None:
        if _has(diff_params, "lambda_rgb") and (not _has(diff_params, "lambda_rgb_edge")):
            _set(diff_params, "lambda_rgb_edge", _get(diff_params, "lambda_rgb", 0.0))
            _pop(diff_params, "lambda_rgb", None)

    # ---------------------------------------------------------
    # build components
    # ---------------------------------------------------------
    cond_uvit = instantiate_from_config(
        cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass)
    )

    model = recurse_instantiate_from_config(
        cfg.model,
        unet=cond_uvit
    )

    diffusion_model = instantiate_from_config(
        cfg.diffusion_model,
        model=model
    )

    train_loader, test_loader = get_loader(cfg)

  
    optimizer = instantiate_from_config(cfg.optimizer, params=diffusion_model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epoch,
        eta_min=cfg.lr_min
    )

    trainer = Trainer(
        diffusion_model,
        train_loader,
        test_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with='wandb',  
        cfg=cfg,
    )

    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)

    trainer.train()
