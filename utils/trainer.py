# utils/trainer.py  (CORREGIDO / EDGE-SAFE)
import glob
import os
from pathlib import Path

import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb
from utils.train_utils import SmoothedValue, set_random_seed

from model.train_val_forward import simple_train_val_forward


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def _get_device_from_model(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _squeeze_b1(x: torch.Tensor) -> torch.Tensor:
    """
    Arregla el caso típico del test_loader:
      image: (B,1,3,H,W) -> (B,3,H,W)
      pred : (B,1,1,H,W) -> (B,1,H,W)
    """
    if torch.is_tensor(x) and x.dim() == 5 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def _ensure_edge_shape(edge: torch.Tensor) -> torch.Tensor:
    """
    Deja edge como (B,1,H,W) si es tensor.
    Acepta (B,1,1,H,W) / (B,H,W) / (B,1,H,W).
    """
    if edge is None or (not torch.is_tensor(edge)):
        return edge

    edge = _squeeze_b1(edge)  # (B,1,H,W)  
    if edge.dim() == 3:  # (B,H,W)
        edge = edge.unsqueeze(1)  # (B,1,H,W)

    if edge.dim() == 4 and edge.shape[1] != 1:
        edge = edge[:, :1, ...]   
    return edge


def _gather_mean_metric(accelerator: Accelerator, local_values: list, device: torch.device) -> float:
    """
    Promedio global correcto (ponderado por cantidad de samples), no promedio por proceso.
    """
    if len(local_values) == 0:
        local_sum = torch.tensor(0.0, device=device)
        local_n = torch.tensor(0.0, device=device)
    else:
        local_sum = torch.tensor(float(np.sum(local_values)), device=device)
        local_n = torch.tensor(float(len(local_values)), device=device)

    packed = torch.stack([local_sum, local_n], dim=0)  # (2,)
    gathered = accelerator.gather(packed)              
    if gathered.dim() == 1:
        total_sum, total_n = gathered[0], gathered[1]
    else:
        total_sum = gathered[:, 0].sum()
        total_n = gathered[:, 1].sum()

    total_n = torch.clamp(total_n, min=1.0)
    return (total_sum / total_n).item()


def cal_mae(gt, res, thresholding, save_to=None, n=None):
    """
    gt: numpy (H,W) in [0,1]
    res: torch tensor (1,H,W) or (H,W) or (1,1,H,W)
    """
    if torch.is_tensor(res):
        if res.dim() == 4:          
            res_t = res
        elif res.dim() == 3:       #  
            res_t = res.unsqueeze(0)
        elif res.dim() == 2:       #  
            res_t = res.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected res dim: {res.shape}")
        res_t = F.interpolate(res_t, size=gt.shape, mode='bilinear', align_corners=False)
        res_t = res_t.squeeze(0).squeeze(0)  #  
        res_t = (res_t - res_t.min()) / (res_t.max() - res_t.min() + 1e-8)
        res_t = (res_t > 0.5).float() if thresholding else res_t
        res_np = res_t.detach().cpu().numpy()
    else:
        res_np = res

    if save_to is not None:
        plt.imsave(os.path.join(save_to, n), res_np, cmap='gray')

    return float(np.sum(np.abs(res_np - gt)) / (gt.shape[0] * gt.shape[1]))


def run_on_seed(func):
    def wrapper(*args, **kwargs):
        seed = np.random.randint(2147483647)
        set_random_seed(0)
        res = func(*args, **kwargs)
        set_random_seed(seed)
        return res
    return wrapper


class Trainer(object):
    def __init__(
        self,
        model,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader = None,
        train_val_forward_fn=simple_train_val_forward,
        gradient_accumulate_every=1,
        optimizer=None,
        scheduler=None,
        train_num_epoch=100,
        results_folder='./results',
        amp=False,
        fp16=False,
        split_batches=True,
        log_with='wandb',
        cfg=None,
    ):
        super().__init__()

        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            log_with='wandb' if log_with else None,
            gradient_accumulation_steps=gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs]
        )

        project_name = getattr(cfg, "project_name", 'ResidualDiffsuion-v7')
        self.accelerator.init_trackers(project_name, config=cfg)

        create_url_shortcut_of_wandb(accelerator=self.accelerator)
        self.logger = create_logger_of_wandb(
            accelerator=self.accelerator,
            rank=not self.accelerator.is_main_process
        )
        self.accelerator.native_amp = amp

        self.model = model
        self.train_val_forward_fn = train_val_forward_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_epoch = train_num_epoch
        self.opt = optimizer
        self.scheduler = scheduler

        if self.accelerator.is_main_process:
            self.results_folder = Path(
                results_folder if results_folder
                else os.path.join(self.accelerator.get_tracker('wandb', unwrap=True).dir, "../")
            )
            self.results_folder.mkdir(exist_ok=True)

        self.cur_epoch = 0

        self.model, self.opt, self.scheduler, self.train_loader, self.test_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, self.train_loader, self.test_loader
        )

    def save(self, epoch, max_to_keep=10):
        if not self.accelerator.is_local_main_process:
            return

        ckpt_files = glob.glob(os.path.join(self.results_folder, 'model-[0-9]*.pt'))
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        ckpt_files_to_delete = ckpt_files[:-max_to_keep]
        for ckpt_file in ckpt_files_to_delete:
            os.remove(ckpt_file)

        data = {
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        save_name = str(self.results_folder / f'model-{epoch}.pt')
        last_save_name = str(self.results_folder / f'model-{epoch}-last.pt')

        if os.path.exists(save_name):
            os.remove(last_save_name) if os.path.exists(last_save_name) else None
            os.rename(save_name, last_save_name)

        torch.save(data, save_name)

    def load(self, resume_path: str = None, pretrained_path: str = None):
        accelerator = self.accelerator
        device = accelerator.device

        if resume_path is not None:
            data = torch.load(resume_path, map_location=device)
            self.cur_epoch = data['epoch']
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

        elif pretrained_path is not None:
            data = torch.load(pretrained_path, map_location=device)
        else:
            raise ValueError('Must specify either resume_path or pretrained_path')

        if self.scheduler is not None:
            for _ in range(self.cur_epoch):
                self.scheduler.step()

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)

    # ---------------------------
    # VALIDATION
    # ---------------------------
    @torch.inference_mode()
    @run_on_seed
    def val(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = _get_device_from_model(model)

        maes = []

        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image = data['image']
            gt = data['gt']
            name = data['name']

            edge = data.get('edge', None)
            edge = _ensure_edge_shape(edge)

            #  
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]

            image = _squeeze_b1(image).to(device)

            if edge is not None and torch.is_tensor(edge):
                edge = edge.to(device)

            out = self.train_val_forward_fn(model, image=image, edge=edge, verbose=False)
            res = out["pred"].detach().cpu()   

            maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, res, name)]

        accelerator.wait_for_everyone()
        mae = _gather_mean_metric(accelerator, maes, device)

        _best_mae = min(_best_mae, mae)
        return mae, _best_mae

    @torch.inference_mode()
    @run_on_seed
    def val_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        def cal_mae_local(gt_np, res_tensor, save_to=None, n=None):
             
            res = res_tensor.detach().cpu().numpy().squeeze()
            if thresholding:
                res = (res > 0.5).astype(np.float32)
            if save_to is not None:
                plt.imsave(os.path.join(save_to, n), res, cmap='gray')
            return float(np.sum(np.abs(res - gt_np)) / (gt_np.shape[0] * gt_np.shape[1]))

        model.eval()
        model = accelerator.unwrap_model(model)
        device = _get_device_from_model(model)

        maes = []

        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image = data['image']
            gt = data['gt']
            name = data['name']

            edge = data.get('edge', None)
            edge = _ensure_edge_shape(edge)

            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]

            image = _squeeze_b1(image).to(device)
            if edge is not None and torch.is_tensor(edge):
                edge = edge.to(device)

            ensem_out = self.train_val_forward_fn(
                model,
                image=image,
                edge=edge,
                time_ensemble=True,
                gt_sizes=[g.shape for g in gt],
                verbose=False
            )
            ensem_res = ensem_out["pred"]   

            maes += [cal_mae_local(g, r, save_to, n) for g, r, n in zip(gt, ensem_res, name)]

        accelerator.wait_for_everyone()
        mae = _gather_mean_metric(accelerator, maes, device)

        _best_mae = min(_best_mae, mae)
        return mae, _best_mae

    @torch.inference_mode()
    @run_on_seed
    def val_batch_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = _get_device_from_model(model)

        maes = []

        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image = data['image']
            gt = data['gt']
            name = data['name']

            edge = data.get('edge', None)
            edge = _ensure_edge_shape(edge)

            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]

            image = _squeeze_b1(image).to(device)
            if edge is not None and torch.is_tensor(edge):
                edge = edge.to(device)

            K = 5
            samples = []
            for _ in range(K):
                out = self.train_val_forward_fn(model, image=image, edge=edge, verbose=False)
                samples.append(out["pred"].detach())  

            batch_res = torch.stack(samples, dim=0).mean(dim=0).cpu()   

            for g, r, n in zip(gt, batch_res, name):
                maes.append(cal_mae(g, r, thresholding, save_to, n))

        accelerator.wait_for_everyone()
        mae = _gather_mean_metric(accelerator, maes, device)

        _best_mae = min(_best_mae, mae)
        return mae, _best_mae

    # ---------------------------
    # TRAIN
    # ---------------------------
    def _forward_from_batch(self, batch: dict, model, **extra_kwargs):

        gt = batch.get("gt", None)
        image = batch.get("image", None)
        seg = batch.get("seg", None)
        edge = batch.get("edge", None)

        # 
        if torch.is_tensor(image):
            image = _squeeze_b1(image)
        if torch.is_tensor(gt):
            gt = _squeeze_b1(gt)
        if torch.is_tensor(seg):
            seg = _squeeze_b1(seg)
        edge = _ensure_edge_shape(edge)

        # 
        passthrough = {}
        for k, v in batch.items():
            if k in ("gt", "image", "seg", "edge", "name", "image_for_post"):
                continue
           
            if torch.is_tensor(v) or isinstance(v, (int, float, bool, str)):
                passthrough[k] = v

        passthrough.update(extra_kwargs)

        return self.train_val_forward_fn(
            model=model,
            gt=gt,
            image=image,
            seg=seg,
            edge=edge,
            **passthrough
        )

    def train(self):
        accelerator = self.accelerator

        last_batch = None

        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch

            self.model.train()
            loss_sm = SmoothedValue(window_size=10)

            with tqdm(total=len(self.train_loader), disable=not accelerator.is_main_process) as pbar:
                for batch in self.train_loader:
                    last_batch = batch

                    with accelerator.autocast(), accelerator.accumulate(self.model):
                        loss = self._forward_from_batch(batch, model=self.model)

                        accelerator.backward(loss)
                        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                        self.opt.step()
                        self.opt.zero_grad()

                    loss_sm.update(loss.item())
                    pbar.set_description(
                        f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm.avg:.4f}({loss_sm.global_avg:.4f})'
                    )
                    accelerator.log({'loss': loss_sm.avg, 'lr': self.opt.param_groups[0]['lr']})
                    pbar.update()

            if self.scheduler is not None:
                self.scheduler.step()

            accelerator.wait_for_everyone()
            loss_sm_gather = accelerator.gather(torch.tensor([loss_sm.global_avg], device=accelerator.device))
            loss_sm_avg = loss_sm_gather.mean().item()
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm_avg:.4f}')

            # Val
            self.model.eval()
            if (epoch + 1) % 1 == 0 or (epoch >= self.train_num_epoch * 0.7):
                mae, best_mae = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} mae: {mae:.4f}({best_mae:.4f})')
                accelerator.log({'mae': mae, 'best_mae': best_mae})
                if mae == best_mae:
                    self.save("best")

            self.save(self.cur_epoch)

            # Visualize (SAFE)
            with torch.inference_mode():
                if accelerator.is_main_process and last_batch is not None:
                    model = accelerator.unwrap_model(self.model)
                    model.eval()

                    out = self._forward_from_batch(last_batch, model=model, verbose=False)
                    # 
                    if isinstance(out, dict) and ("pred" in out):
                        pred = out["pred"]
                        img = out.get("image", None)
                        gt = out.get("gt", None)
                        edge = last_batch.get("edge", None)
                        edge = _ensure_edge_shape(edge)

                        def _to_img2d(x):
                            if x is None or (not torch.is_tensor(x)):
                                return None
                            x = _squeeze_b1(x)
                            if x.dim() == 4:  
                                x0 = x[0]
                                if x0.shape[0] == 3:
                                    
                                    x0 = x0.detach().cpu().float()
                                    x0 = (x0 - x0.min()) / (x0.max() - x0.min() + 1e-8)
                                    return x0.permute(1, 2, 0).numpy()
                                else:
                                    x0 = x0[0].detach().cpu().float()
                                    x0 = (x0 - x0.min()) / (x0.max() - x0.min() + 1e-8)
                                    return x0.numpy()
                            return None

                        pred_img = _to_img2d(pred)
                        rgb_img = _to_img2d(img)
                        gt_img = _to_img2d(gt)
                        edge_img = _to_img2d(edge) if torch.is_tensor(edge) else None

                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                log_payload = {}
                                if rgb_img is not None:
                                    log_payload["vis/rgb"] = wandb.Image(rgb_img)
                                if pred_img is not None:
                                    log_payload["vis/pred"] = wandb.Image(pred_img)
                                if gt_img is not None:
                                    log_payload["vis/gt"] = wandb.Image(gt_img)
                                if edge_img is not None:
                                    log_payload["vis/edge"] = wandb.Image(edge_img)

                                if len(log_payload) > 0:
                                    tracker.log(log_payload)

            accelerator.wait_for_everyone()

        self.logger.info('training complete')
        accelerator.end_training()
