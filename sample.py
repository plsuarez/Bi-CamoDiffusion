import sys
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.train_utils import set_random_seed
from utils import init_env  # noqa: F401
from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from utils.trainer import Trainer

set_random_seed(7)


def get_loader(cfg):
    cod10k_test_dataset = instantiate_from_config(cfg.test_dataset.COD10K)
    camo_test_dataset = instantiate_from_config(cfg.test_dataset.CAMO)
    chameleon_test_dataset = instantiate_from_config(cfg.test_dataset.CHAMELEON)
    nc4k_test_dataset = instantiate_from_config(cfg.test_dataset.NC4K)

    cod10k_test_loader = DataLoader(cod10k_test_dataset, batch_size=cfg.batch_size, collate_fn=collate)
    camo_test_loader = DataLoader(camo_test_dataset, batch_size=cfg.batch_size, collate_fn=collate)
    chameleon_test_loader = DataLoader(chameleon_test_dataset, batch_size=cfg.batch_size, collate_fn=collate)
    nc4k_test_loader = DataLoader(nc4k_test_dataset, batch_size=cfg.batch_size, collate_fn=collate)

    return cod10k_test_loader, camo_test_loader, chameleon_test_loader, nc4k_test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)
    parser.add_argument('--target_dataset', nargs='+', type=str, default=['CAMO', 'COD10K', 'CHAMELEON', 'NC4K'])
    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')

    # NEW: enable edge at eval time (optional flag from CLI)
    parser.add_argument('--use_edge', action='store_true', help='Enable edge conditioning during sampling/validation')

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'

    # Hack config here.
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    # NEW: force diffusion_model to use edges if requested
    if cfg.use_edge:
        # diffusion_model has params.use_edge in your corrected diffusion class
        if hasattr(cfg.diffusion_model, "params"):
            cfg.diffusion_model.params.use_edge = True

        # optional threshold if you keep it in params
        if hasattr(cfg.diffusion_model.params, "edge_threshold"):
            # keep your default unless you want to override
            pass

    cod10k_test_loader, camo_test_loader, chameleon_test_loader, nc4k_test_loader = get_loader(cfg)

    cond_uvit = instantiate_from_config(
        cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass)
    )

    model = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)

    diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model)

    # NEW: also set runtime flag, in case cfg didn't propagate exactly
    if cfg.use_edge and hasattr(diffusion_model, "use_edge"):
        diffusion_model.use_edge = True

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)

    cod10k_test_loader, camo_test_loader, chameleon_test_loader, nc4k_test_loader = trainer.accelerator.prepare(
        cod10k_test_loader, camo_test_loader, chameleon_test_loader, nc4k_test_loader
    )

    dataset_map = {
        'CAMO': camo_test_loader,
        'COD10K': cod10k_test_loader,
        'CHAMELEON': chameleon_test_loader,
        'NC4K': nc4k_test_loader,
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()} Your input: {cfg.target_dataset}'

    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()

        mask_path = Path(cfg.test_dataset.CAMO.params.image_root).parent.parent
        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)

        if cfg.batch_ensemble:
            mae, _ = trainer.val_batch_ensemble(
                model=trainer.model,
                test_data_loader=dataset,
                accelerator=trainer.accelerator,
                thresholding=False,
                save_to=save_to
            )
        elif cfg.time_ensemble:
            mae, _ = trainer.val_time_ensemble(
                model=trainer.model,
                test_data_loader=dataset,
                accelerator=trainer.accelerator,
                thresholding=False,
                save_to=save_to
            )
        else:
            mae, _ = trainer.val(
                model=trainer.model,
                test_data_loader=dataset,
                accelerator=trainer.accelerator,
                thresholding=False,
                save_to=save_to
            )

        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')

        if trainer.accelerator.is_main_process:
            from utils.eval import eval
            eval_score = eval(mask_path=mask_path, pred_path=cfg.results_folder, dataset_name=dataset_name)

        trainer.accelerator.wait_for_everyone()
