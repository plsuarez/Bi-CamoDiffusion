import torch
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from torch import nn

from denoising_diffusion_pytorch.simple_diffusion import *


# ============================================================
# Edge helpers 
# ============================================================
def error_edge_losses_multiscale(self, pred_prob, gt_prob, edge01):
     
    total = pred_prob.new_tensor(0.0)

    # Early exit  
    if (self.lambda_gt_edge <= 0.0) and (self.lambda_rgb_edge <= 0.0):
        return total
    if (self.lambda_rgb_edge > 0.0) and (edge01 is None):
        
        return total

    scales = list(self.multiscale) if self.multiscale is not None else [1.0]
    weights = list(self.multiscale_w) if self.multiscale_w is not None else [1.0]

    
    if len(weights) < len(scales):
        weights = weights + [weights[-1]] * (len(scales) - len(weights))
    elif len(weights) > len(scales):
        weights = weights[:len(scales)]

    
    w = torch.tensor(weights, device=pred_prob.device, dtype=pred_prob.dtype)
    w = w / (w.sum() + 1e-8)

    H, W = pred_prob.shape[-2:]

    for s, ws in zip(scales, w):
        if s == 1.0:
            p, g = pred_prob, gt_prob
            e = edge01
        else:
            nh = max(1, int(round(H * float(s))))
            nw = max(1, int(round(W * float(s))))
            size = (nh, nw)

            
            if s < 1.0:
                p = F.interpolate(pred_prob, size=size, mode="area")
                g = F.interpolate(gt_prob,   size=size, mode="area")
            else:
                p = F.interpolate(pred_prob, size=size, mode="bilinear", align_corners=False)
                g = F.interpolate(gt_prob,   size=size, mode="bilinear", align_corners=False)

             
            if edge01 is not None:
                if s < 1.0:
                    e = F.interpolate(edge01, size=size, mode="area")
                else:
                    e = F.interpolate(edge01, size=size, mode="nearest", align_corners=False)
                     
            else:
                e = None

        if self.lambda_gt_edge > 0.0:
            total = total + ws * self.lambda_gt_edge * (sobel_mag(p) - sobel_mag(g)).abs().mean()

        if self.lambda_rgb_edge > 0.0 and e is not None:
            total = total + ws * self.lambda_rgb_edge * (sobel_mag(p) - e).abs().mean()

    return total

def _edge_to_01(edge: torch.Tensor) -> torch.Tensor:
    """
    Convert edge tensor to float in [0,1].
    Accepts:
      - uint8 [0..255]
      - float [0..1]
      - float [-1..1]
    """
    if edge is None:
        return None
    if not torch.is_floating_point(edge):
        edge = edge.float()

    e_max = float(edge.amax().detach().cpu())
    if e_max > 1.0 + 1e-6:
        edge = edge / 255.0

    e_min = float(edge.amin().detach().cpu())
    if e_min < -1e-6:
        edge = (edge + 1.0) / 2.0

    return edge.clamp(0.0, 1.0)

def sobel_mag(x01: torch.Tensor) -> torch.Tensor:
    """
    Sobel magnitude for (B,1,H,W) in [0,1]
    Parameter-free, fixed kernels.
    """
    if x01.dim() == 3:
        x01 = x01.unsqueeze(1)
    if x01.shape[1] != 1:
        x01 = x01.mean(dim=1, keepdim=True)

    x01 = x01.clamp(0.0, 1.0)

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=x01.device, dtype=x01.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=x01.device, dtype=x01.dtype).view(1, 1, 3, 3)

    gx = F.conv2d(x01, kx, padding=1)
    gy = F.conv2d(x01, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


# ============================================================
# Building blocks 
# ============================================================

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

        class LayerNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

            def forward(self, x):
                eps = 1e-5 if x.dtype == torch.float32 else 1e-3
                var = torch.var(x, dim=1, unbiased=False, keepdim=True)
                mean = torch.mean(x, dim=1, keepdim=True)
                return (x - mean) * (var + eps).rsqrt() * self.g

        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# ============================================================
# CondUnetWrapper:  
# ============================================================

class CondUnetWrapper(nn.Module):
    def __init__(
        self,
        unet,
        feature_exactor,
        translayer=None,
        *,
         
        use_edge_inj: bool = False,
        edge_inj_strength: float = 0.10,   
        edge_inj_filter: bool = False,      
    ):
        super().__init__()
        self.feature_exactor = feature_exactor
        self.unet = unet
        self.translayer = translayer

        self.use_edge_inj = bool(use_edge_inj)
        self.edge_inj_strength = float(edge_inj_strength)
        self.edge_inj_filter = bool(edge_inj_filter)

         
        k = torch.tensor([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=torch.float32)   
        self.register_buffer("_K_fixed", k.view(1, 1, 3, 3), persistent=False)

    def _pi_expand(self, edge01: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:

        e = F.interpolate(edge01, size=(H, W), mode="bilinear", align_corners=False)

        if self.edge_inj_filter:
            e = F.conv2d(e, self._K_fixed.to(device=e.device, dtype=e.dtype), padding=1)
            e = e.abs()

        e = e.repeat(1, C, 1, 1)
        return e

    def extract_features(self, cond_img, edge=None):
        features = self.feature_exactor(cond_img)

     
        if self.use_edge_inj and edge is not None:
            e = edge
            if e.dim() == 3:
                e = e.unsqueeze(1)
            if e.shape[1] != 1:
                e = e.mean(dim=1, keepdim=True)
            e = e.clamp(0.0, 1.0).detach()

            if isinstance(features, (list, tuple)):
                feats = list(features)
                f1 = feats[0]
                inj = self._pi_expand(e, f1.shape[1], f1.shape[2], f1.shape[3])
                feats[0] = f1 + self.edge_inj_strength * inj
                features = tuple(feats) if isinstance(features, tuple) else feats
            else:
                f1 = features
                inj = self._pi_expand(e, f1.shape[1], f1.shape[2], f1.shape[3])
                features = f1 + self.edge_inj_strength * inj

        features = self.translayer(features) if self.translayer is not None else features
        return features

    def sample_unet(self, x, times, conditioning_features, edge=None):
        
        return self.unet(x, times, conditioning_features)

    def forward(self, x, times, cond_img, edge=None):
        conditioning_features = self.extract_features(cond_img, edge=edge)
        return self.sample_unet(x, times, conditioning_features, edge=edge)


# ============================================================
# CondUViT  
# ============================================================

class CondUViT(UViT):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), downsample_factor=2, channels=3,
                 out_channels=None, vit_depth=6, vit_dropout=0.2, attn_dim_head=32, attn_heads=4, ff_mult=4,
                 resnet_block_groups=8, learned_sinusoidal_dim=16, init_img_transform: callable = None,
                 final_img_itransform: callable = None, patch_size=1, dual_patchnorm=False, conditioning_klass=None,
                 use_condtionning=(True, True, True, True), condition_dims=None, condition_sizes=None,
                 skip_connect_condition_fmaps=False):
        super().__init__(dim, init_dim, out_dim, dim_mults, downsample_factor, channels, out_channels, vit_depth,
                         vit_dropout,
                         attn_dim_head, attn_heads, ff_mult, resnet_block_groups, learned_sinusoidal_dim,
                         init_img_transform, final_img_itransform, patch_size, dual_patchnorm)
        assert conditioning_klass is not None, \
            "Conditioning class must be provided, which is a class that can be instantiated with fmap_size and dim_in"
        assert len(use_condtionning) == len(condition_dims) == len(condition_sizes), \
            "Conditioning parameters must be of the same length"

        self.skip_connect_condition_fmaps = skip_connect_condition_fmaps
        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        resnet_block = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        num_resolutions = len(in_out)
        assert num_resolutions + 1 == len(use_condtionning), \
            "Condition parameter have an extra original size feature map"

        self.conditioners = nn.ModuleList([])
        for ind, (use_cond, cond_dim, cond_size) in enumerate(zip(use_condtionning, condition_dims, condition_sizes)):
            if use_cond:
                self.conditioners.append(conditioning_klass(cond_size, cond_dim))
            else:
                self.conditioners.append(None)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out, factor=factor)
            ]))

        for ind, ((dim_in, dim_out), factor, cond_dim) \
                in enumerate(zip(reversed(in_out), reversed(downsample_factor),
                                 reversed(condition_dims[:-1]))):
            skip_connect_dim = cond_dim if self.skip_connect_condition_fmaps else 0
            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, factor=factor),
                resnet_block(dim_in * 2 + skip_connect_dim, dim_in, time_emb_dim=time_dim),
                resnet_block(dim_in * 2 + skip_connect_dim, dim_in, time_emb_dim=time_dim),
                LinearAttention(dim_in),
            ]))

    def forward(self, x, times, cond):
        skip_connect_c = self.skip_connect_condition_fmaps
        assert len(cond) == len(self.conditioners)

        x = self.init_img_transform(x)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(times)
        h = []

        for (block1, block2, attn, downsample), cond_feature, conditioner in zip(self.downs, cond, self.conditioners):
            x = block1(x, t)
            h.append([x, cond_feature] if skip_connect_c else [x])

            x = block2(x, t)
            x = attn(x)
            x = (x + conditioner(x, cond_feature)) if conditioner is not None else x

            h.append([x, cond_feature] if skip_connect_c else [x])
            x = downsample(x)

        x = (x + self.conditioners[-1](x, cond[-1])) if self.conditioners[-1] is not None else x

        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        x = self.vit(x, t)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')

        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)
            x = torch.cat((x, *h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, *h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = self.unpatchify(x)
        return self.final_img_itransform(x)

import torch
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm

from denoising_diffusion_pytorch.simple_diffusion import (
    GaussianDiffusion,
    logsnr_schedule_cosine,
    default,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    right_pad_dims_to,
    sqrt,
    expm1,
)

#  
try:
    from model.loss import sanitize_edge, sobel_mag, ual_loss
except Exception:
    # fallback  
    def nonsanitize_edge(edge, threshold=0.25, smooth=True, detach=True):
        if edge is None:
            return None
        if edge.dim() == 3:
            edge = edge.unsqueeze(1)
        edge = edge.float()
        if edge.max() > 1.5:
            edge = edge / 255.0
        edge = edge.clamp(0.0, 1.0)
        if smooth:
            edge = F.avg_pool2d(edge, 3, 1, 1)
        if threshold is not None and threshold > 0:
            denom = max(1e-6, 1.0 - float(threshold))
            edge = ((edge - float(threshold)) / denom).clamp(0.0, 1.0)
        return edge.detach() if detach else edge

    def nonsobel_mag(x, eps=1e-6):
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        kx = torch.tensor([[1, 0, -1],[2,0,-2],[1,0,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
        ky = torch.tensor([[1, 2,  1],[0,0, 0],[-1,-2,-1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + eps)


class CondGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=1,
        extra_channels=0,
        cond_channels=3,
        pred_objective='v',
        loss_type='l2',
        noise_schedule=logsnr_schedule_cosine,
        noise_d=None,
        noise_d_low=None,
        noise_d_high=None,
        num_sample_steps=500,
        clip_sample_denoised=True,

        # paper losses  
        lambda_gt_edge: float = 0.0,      
        lambda_rgb_edge: float = 0.0,     
	lambda_ual: float = 0.0,         
	ual_gamma: float = 1.0,           
        #  
        lambda_rgb: float = None,

        edge_threshold: float = 0.25,
        edge_smooth: bool = True,

        #  
        multiscale=(1.0,),
        multiscale_w=(1.0,),
        **kwargs
    ):
        # -----------------------------
        # 1 
        # -----------------------------
        if lambda_rgb is not None and float(lambda_rgb_edge) == 0.0:
            lambda_rgb_edge = float(lambda_rgb)

        
        if noise_schedule is None:
            noise_schedule = logsnr_schedule_cosine
         
        if isinstance(noise_schedule, str):
            from utils.import_utils import get_obj_from_str
            noise_schedule = get_obj_from_str(noise_schedule)

        super().__init__(
            model,
            image_size=image_size,
            channels=channels,
            pred_objective=pred_objective,
            noise_schedule=noise_schedule,
            noise_d=noise_d,
            noise_d_low=noise_d_low,
            noise_d_high=noise_d_high,
            num_sample_steps=num_sample_steps,
            clip_sample_denoised=clip_sample_denoised
        )

        #  
        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                loss_type = get_obj_from_str(loss_type)
            except:
                raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        self.loss_type = loss_type
        self.extra_channels = extra_channels
        self.cond_channels = cond_channels

        #  
        self.lambda_gt_edge = float(lambda_gt_edge)
        self.lambda_rgb_edge = float(lambda_rgb_edge)
        self.edge_threshold = float(edge_threshold)
        self.edge_smooth = bool(edge_smooth)
        self.lambda_ual = float(lambda_ual)
        self.ual_gamma = float(ual_gamma)

        self.multiscale = tuple([float(s) for s in (multiscale or (1.0,))])
        self.multiscale_w = tuple([float(w) for w in (multiscale_w or (1.0,))])

        self.history = []

    # -----------------------------
    # helpers
    # -----------------------------
    def _prep_edge(self, edge):
        if edge is None:
            return None
        return sanitize_edge(edge, threshold=self.edge_threshold, smooth=self.edge_smooth, detach=True)

    def _maybe_call_with_edge(self, fn, *args, edge=None, **kwargs):
        """Llama fn(..., edge=edge)  """
        if edge is None:
            return fn(*args, **kwargs)
        try:
            return fn(*args, edge=edge, **kwargs)
        except TypeError as e:
            if "edge" in str(e):
                return fn(*args, **kwargs)
            raise

    # -----------------------------
    # forward / losses
    # -----------------------------
    def forward(self, img, cond_img, seg=None, extra_cond=None, edge=None, *args, **kwargs):
        b, channels, h, w = img.shape
        assert channels == self.channels
        assert h == w == self.image_size
        assert cond_img.shape[1] == self.cond_channels

        img = normalize_to_neg_one_to_one(img)
        seg = normalize_to_neg_one_to_one(seg) if seg is not None else None

        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        times = torch.zeros((b,), device=self.device).float().uniform_(0, 1)

        edge01 = self._prep_edge(edge)
        return self.p_losses(img, times, cond_img, seg, extra_cond, edge01=edge01, *args, **kwargs)

    def _consensus_time_ensemble(self, last_k: int = 10, weighted: bool = True):
        """
        CTE:  
         
        Return:
        p_cte: prob (B,1,H,W)  [0,1]
        m_cte: mask (B,1,H,W)  {0,1}
        """
        if not hasattr(self, "history") or len(self.history) == 0:
            return None, None

        hist = self.history
        if last_k is not None and last_k > 0:
            hist = hist[-last_k:]

        xs = torch.stack(hist, dim=0)                 
        probs = ((xs + 1.0) / 2.0).clamp(0.0, 1.0)    

        if not weighted:
            p = probs.mean(dim=0)
        else:
            K = probs.shape[0]
            w = torch.linspace(1.0, 2.0, K, device=probs.device, dtype=probs.dtype)
            w = w / (w.sum() + 1e-8)
            p = (probs * w.view(K, 1, 1, 1, 1)).sum(dim=0)

        m = (p >= 0.5).float()
        return p, m
    def _ms_loss_logits(self, pred_logits: torch.Tensor, gt01: torch.Tensor) -> torch.Tensor:

      	#"""
        scales = self.multiscale if self.multiscale is not None else (1.0,)
        weights = self.multiscale_w if self.multiscale_w is not None else (1.0,)
        if len(weights) != len(scales):
         
            if len(weights) == 1:
                weights = (weights[0],) * len(scales)
            else:
                raise ValueError(f"multiscale_w ({len(weights)}) != multiscale ({len(scales)})")

        # normalize
        w = torch.tensor(weights, device=pred_logits.device, dtype=pred_logits.dtype)
        w = w / (w.sum() + 1e-8)

        B, C, H, W = pred_logits.shape
        total = pred_logits.new_tensor(0.0)

        for s, ws in zip(scales, w):
            s = float(s)

            if s == 1.0:
                z_s = pred_logits
                y_s = gt01
            else:
                nh = max(1, int(round(H * s)))
                nw = max(1, int(round(W * s)))
                size = (nh, nw)

                # logits:  
                z_s = F.interpolate(pred_logits, size=size, mode="bilinear", align_corners=False)

                # GT: nearest  
                y_s = F.interpolate(gt01, size=size, mode="nearest")

            #  L_fs (structure_loss )
            if not callable(self.loss_type):
                raise ValueError("Para L_ms necesitas loss_type callable (ej: model.loss.structure_loss).")
            loss_s = self.loss_type(z_s, y_s)

            #  L_gt-edge = |S(sigmoid(z)) - S(y)|
            if self.lambda_gt_edge > 0.0:
                p_s = torch.sigmoid(z_s).clamp(0.0, 1.0)
                loss_gt_edge = (sobel_mag(p_s) - sobel_mag(y_s)).abs().mean()
                loss_s = loss_s + self.lambda_gt_edge * loss_gt_edge

            #  L_ual
            if self.lambda_ual > 0.0:
                loss_s = loss_s + self.lambda_ual * ual_loss(z_s, y_s, gamma=self.ual_gamma)

            total = total + ws * loss_s

        return total
    def p_losses(self, x_start, times, cond_img,
             seg=None, extra_cond=None, noise=None,
             edge=None, edge01=None, *args, **kwargs):

        # --- 0) noise by default ---
        noise = default(noise, lambda: torch.randn_like(x_start))

        # --- compatibility:  edge01  ---
        if edge is None and edge01 is not None:
            edge = edge01

        # (opcional) limpia kwargs por si viene duplicado 
        kwargs.pop("edge01", None)
        kwargs.pop("edge", None)

        # --- 1) define x0 real (diffusion en [-1,1]) ---
        x0 = seg if seg is not None else x_start  # [-1,1]

        # --- 2) forward diffusion ---
        x, log_snr = self.q_sample(x_start=x0, times=times, noise=noise)

        # --- 3) model ---
        if extra_cond is None:
            model_in = x
        else:
            model_in = torch.cat([x, extra_cond], dim=1)

        #  
        model_out = self._maybe_call_with_edge(self.model, model_in, log_snr, cond_img, edge=edge)

        # ---   base loss + pred reconstruction ---
        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha = padded_log_snr.sigmoid().sqrt()
            sigma = (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x0

            base_pred = model_out
            base_loss = self.loss_type(base_pred, target) if callable(self.loss_type) and self.loss_type not in ['l2','l1','l1+l2','mean(l1, l2)'] else None
            if base_loss is None:
                if self.loss_type == 'l2':
                    base_loss = F.mse_loss(base_pred, target)
                elif self.loss_type == 'l1':
                    base_loss = F.l1_loss(base_pred, target)
                elif self.loss_type == 'l1+l2':
                    base_loss = F.mse_loss(base_pred, target) + F.l1_loss(base_pred, target)
                elif self.loss_type == 'mean(l1, l2)':
                    base_loss = (F.mse_loss(base_pred, target) + F.l1_loss(base_pred, target)) / 2
                else:
                    base_loss = self.loss_type(base_pred, target)

            x0_pred = (alpha * x - sigma * model_out).clamp(-1.0, 1.0)
            pred_prob = ((x0_pred + 1.0) / 2.0).clamp(0.0, 1.0)
            gt_prob   = ((x0     + 1.0) / 2.0).clamp(0.0, 1.0)

        elif self.pred_objective == 'eps':
            target = noise

            base_pred = model_out
            base_loss = self.loss_type(base_pred, target) if callable(self.loss_type) and self.loss_type not in ['l2','l1','l1+l2','mean(l1, l2)'] else None
            if base_loss is None:
                if self.loss_type == 'l2':
                    base_loss = F.mse_loss(base_pred, target)
                elif self.loss_type == 'l1':
                    base_loss = F.l1_loss(base_pred, target)
                elif self.loss_type == 'l1+l2':
                    base_loss = F.mse_loss(base_pred, target) + F.l1_loss(base_pred, target)
                elif self.loss_type == 'mean(l1, l2)':
                    base_loss = (F.mse_loss(base_pred, target) + F.l1_loss(base_pred, target)) / 2
                else:
                    base_loss = self.loss_type(base_pred, target)

            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha = padded_log_snr.sigmoid().sqrt()
            sigma = (-padded_log_snr).sigmoid().sqrt()
            x0_pred = ((x - sigma * model_out) / (alpha + 1e-8)).clamp(-1.0, 1.0)

            pred_prob = ((x0_pred + 1.0) / 2.0).clamp(0.0, 1.0)
            gt_prob   = ((x0     + 1.0) / 2.0).clamp(0.0, 1.0)

        elif self.pred_objective == 'x0':
            # LOGITS
            pred_logits = model_out

            # GT en [0,1]
            gt01 = ((x0 + 1.0) / 2.0).clamp(0.0, 1.0)
            if gt01.dim() == 3:
                gt01 = gt01.unsqueeze(1)

            # --- L_ms   ---
            base_loss = self._ms_loss_logits(pred_logits, gt01)

            # ---   RGB-edge (  ) ---
            if (edge is not None) and (self.lambda_rgb_edge > 0.0):
                #  
                e = edge
                if e.dim() == 3:
                    e = e.unsqueeze(1)
                if e.shape[1] != 1:
                    e = e.mean(dim=1, keepdim=True)

                #  
                if e.shape[-2:] != pred_logits.shape[-2:]:
                    e = F.interpolate(e, size=pred_logits.shape[-2:], mode="nearest")

                pred_prob = torch.sigmoid(pred_logits).clamp(0.0, 1.0)
                rgb_edge_loss = (sobel_mag(pred_prob) - e).abs().mean()
                base_loss = base_loss + self.lambda_rgb_edge * rgb_edge_loss

            return base_loss

        
    # -----------------------------
    # sampling (edge optional)
    # -----------------------------
    @torch.no_grad()
    def sample(
        self,
        cond_img,
        extra_cond=None,
        edge=None,
        verbose=True,
        *,
        return_cte: bool = False,
        cte_last_k: int = 10,
        cte_weighted: bool = True,
        cte_as_mask: bool = False,
    ):
        b, c, h, w = cond_img.shape
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        edge01 = self._prep_edge(edge)
        return self.p_sample_loop(
            (b, self.channels, self.image_size, self.image_size),
            cond_img,
            extra_cond=extra_cond,
            edge01=edge01,
            verbose=verbose,
            return_cte=return_cte,
            cte_last_k=cte_last_k,
            cte_weighted=cte_weighted,
            cte_as_mask=cte_as_mask,
        )


    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond_img,
        extra_cond,
        edge01=None,
        verbose=True,
        *,
        return_cte: bool = False,
        cte_last_k: int = 10,
        cte_weighted: bool = True,
        cte_as_mask: bool = False,
    ):
        self.history = []
        img = torch.randn(shape, device=self.device)

        conditioning_features = self._maybe_call_with_edge(self.model.extract_features, cond_img, edge=edge01)

        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                    disable=not verbose):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, conditioning_features, extra_cond, times, times_next, edge01=edge01)

        #  
        img.clamp_(-1., 1.)
        img01 = unnormalize_to_zero_to_one(img)

        # CTE (op)
        if return_cte:
            p_cte, m_cte = self._consensus_time_ensemble(last_k=cte_last_k, weighted=cte_weighted)
            if p_cte is not None:
                return m_cte if cte_as_mask else p_cte

        return img01
   
    @torch.no_grad()
    def p_sample(self, x, cond, extra_cond, time, time_next, edge01=None):
        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, extra_cond=extra_cond, time=time,
                                                          time_next=time_next, edge01=edge01)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    def p_mean_variance(self, x, cond, extra_cond, time, time_next, edge01=None):
        log_snr      = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        # robust: alpha/sigma 
        padded_log_snr      = right_pad_dims_to(x, log_snr)
        padded_log_snr_next = right_pad_dims_to(x, log_snr_next)

        squared_alpha      = padded_log_snr.sigmoid()
        squared_alpha_next = padded_log_snr_next.sigmoid()
        squared_sigma      = (-padded_log_snr).sigmoid()
        squared_sigma_next = (-padded_log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])

        model_in = torch.cat([x, extra_cond], dim=1)
        pred = self._maybe_call_with_edge(self.model.sample_unet, model_in, batch_log_snr, cond, edge=edge01)

        if self.pred_objective == 'v':
           x0_pred = alpha * x - sigma * pred
        elif self.pred_objective == 'eps':
            x0_pred = (x - sigma * pred) / (alpha + 1e-8)
        elif self.pred_objective == 'x0':
            pred_prob = torch.sigmoid(pred)                 # [0,1]
            x0_pred = (pred_prob * 2.0 - 1.0)  
            
        else:
            raise ValueError(f"Unknown pred_objective: {self.pred_objective}")

        x0_pred = x0_pred.clamp(-1., 1.)
        self.history.append(x0_pred)

        model_mean = alpha_next * (x * (1 - c) / (alpha + 1e-8) + c * x0_pred)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance
class ResCondGaussianDiffusion(CondGaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super(ResCondGaussianDiffusion, self).__init__(*args, **kwargs)

    def forward(self, img, cond_img, seg=None, extra_cond=None, edge=None, *args, **kwargs):
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]
        assert channels == self.channels
        assert h == w == self.image_size
        assert cond_channels == self.cond_channels

        seg = normalize_to_neg_one_to_one(seg) if seg is not None else None
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        times = torch.zeros((img.shape[0],), device=self.device).float().uniform_(0, 1)

        edge01 = self._prep_edge(edge)
        return self.p_losses(img, times, cond_img, seg, extra_cond, edge01=edge01, *args, **kwargs)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_img, extra_cond, edge01=None, verbose=True):
        self.history = []
        img = torch.randn(shape, device=self.device)
        conditioning_features = self.model.extract_features(cond_img, edge=edge01)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                      disable=not verbose):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, conditioning_features, extra_cond, times, times_next, edge01=edge01)

        img.clamp_(-1., 1.)
        return img
