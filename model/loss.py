import torch
import torch.nn as nn
import torch.nn.functional as F

# loss.py
_sobel_cache = {}

def _get_sobel_kernels(device, dtype):
    key = (device.type, device.index, dtype)
    if key in _sobel_cache:
        return _sobel_cache[key]

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)
    _sobel_cache[key] = (kx, ky)
    return kx, ky

def sobel_mag(x01: torch.Tensor):
    if x01.dim() == 3:
        x01 = x01.unsqueeze(1)
    if x01.shape[1] != 1:
        x01 = x01.mean(dim=1, keepdim=True)

    x01 = x01.clamp(0.0, 1.0)
    kx, ky = _get_sobel_kernels(x01.device, x01.dtype)

    gx = F.conv2d(x01, kx, padding=1)
    gy = F.conv2d(x01, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return mag

import torch
import torch.nn.functional as F

def sanitize_edge(edge: torch.Tensor, threshold=0.25, smooth=True, detach=True):
    if edge is None:
        return None

    
    if edge.dim() == 3:
        edge = edge.unsqueeze(1)
    if edge.shape[1] != 1:
        edge = edge.mean(dim=1, keepdim=True)

     
    e = edge.float().clamp(0.0, 1.0)

    if smooth:
        e = F.avg_pool2d(e, 3, 1, 1)

    if threshold is not None and threshold > 0:
         
        den = max(1e-6, 1.0 - float(threshold))
        e = ((e - float(threshold)) / den).clamp(0.0, 1.0)

    return e.detach() if detach else e


# ============================================================
#
# ============================================================

def focal_structure_loss(pred_logits: torch.Tensor,
                         mask: torch.Tensor,
                         gamma: float = 2.0) -> torch.Tensor:
     
    mask = _to_01(mask)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

     
    weit = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred_logits, mask, reduction='none')
    prob = torch.sigmoid(pred_logits)

 
    pt = prob * mask + (1.0 - prob) * (1.0 - mask)
    focal = (1.0 - pt).pow(gamma)

    fbce = focal * bce
    fbce = (weit * fbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)

    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

    return (fbce + wiou).mean()

def gt_edge_loss(pred_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
 
    mask = _to_01(mask)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    prob = torch.sigmoid(pred_logits).clamp(0.0, 1.0)
    return (sobel_mag(prob) - sobel_mag(mask)).abs().mean()

def cal_ual(seg_logits: torch.Tensor, seg_gts: torch.Tensor) -> torch.Tensor:
 
    if seg_gts.dim() == 3:
        seg_gts = seg_gts.unsqueeze(1)
    seg_gts = _to_01(seg_gts)

    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1.0 - (2.0 * sigmoid_x - 1.0).abs().pow(2)
    return loss_map.mean()

def rgb_edge_consistency_loss(pred_logits: torch.Tensor,
                              edge: torch.Tensor,
                              edge_threshold: float = 0.25,
                              edge_smooth: bool = True) -> torch.Tensor:
 
    prob = torch.sigmoid(pred_logits).clamp(0.0, 1.0)

    edge01 = sanitize_edge(edge, threshold=edge_threshold, smooth=edge_smooth, detach=True)
    edge01 = F.interpolate(edge01, size=prob.shape[-2:], mode="bilinear", align_corners=False)

    return (sobel_mag(prob) - edge01).abs().mean()

def focal_structure_edge_ual_multiscale_loss(
        pred_logits: torch.Tensor,
        mask: torch.Tensor,
        edge: torch.Tensor = None,
       
        multiscale=(1.0,),
        multiscale_w=(1.0,),
       
        lambda_gt_edge: float = 0.0,
        lambda_ual: float = 0.0,
        lambda_rgb_edge: float = 0.0,   
        
        focal_gamma: float = 2.0,
        edge_threshold: float = 0.25,
        edge_smooth: bool = True,
) -> torch.Tensor:
 
    if len(multiscale) != len(multiscale_w):
        raise ValueError("multiscale y multiscale_w deben tener la misma longitud")

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = _to_01(mask)

    total = 0.0
    wsum = 0.0

    B, _, H, W = mask.shape

    for s, w in zip(multiscale, multiscale_w):
        s = float(s)
        w = float(w)
        if s == 1.0:
            z_s = pred_logits
            y_s = mask
            e_s = edge
        else:
            h_s = max(1, int(round(H * s)))
            w_s = max(1, int(round(W * s)))
            z_s = F.interpolate(pred_logits, size=(h_s, w_s), mode="bilinear", align_corners=False)
            y_s = F.interpolate(mask,       size=(h_s, w_s), mode="bilinear", align_corners=False)
            e_s = F.interpolate(edge,       size=(h_s, w_s), mode="bilinear", align_corners=False) if edge is not None else None

        loss_s = focal_structure_loss(z_s, y_s, gamma=focal_gamma)

        if lambda_gt_edge > 0.0:
            loss_s = loss_s + float(lambda_gt_edge) * gt_edge_loss(z_s, y_s)

        if lambda_ual > 0.0:
            loss_s = loss_s + float(lambda_ual) * cal_ual(z_s, y_s)

        if lambda_rgb_edge > 0.0 and e_s is not None:
            loss_s = loss_s + float(lambda_rgb_edge) * rgb_edge_consistency_loss(
                z_s, e_s, edge_threshold=edge_threshold, edge_smooth=edge_smooth
            )

        total = total + w * loss_s
        wsum += w

    total = total / max(1e-6, wsum)
    return total

 

def structure_loss(pred, mask):
    #mask = _to_01(mask)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)

    predp = torch.sigmoid(pred)
    inter = ((predp * mask) * weit).sum(dim=(2, 3))
    union = ((predp + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1 + 1e-6)

    return (wbce + wiou).mean()
import torch
import torch.nn.functional as F

def _to_01(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if not torch.is_floating_point(x):
        x = x.float()

    x_max = float(x.amax().detach().cpu())
    if x_max > 1.0 + 1e-6:
        x = x / 255.0

    x_min = float(x.amin().detach().cpu())
    if x_min < -1e-6:
        x = (x + 1.0) / 2.0

    return x.clamp(0.0, 1.0)


def ual_loss(pred_logits: torch.Tensor, mask: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    mask = _to_01(mask)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    p = torch.sigmoid(pred_logits).clamp(0.0, 1.0)
    u = (1.0 - (2.0 * p - 1.0).abs()).clamp(0.0, 1.0)   

    if gamma is not None and gamma != 1.0:
        u = u.pow(float(gamma))

     
    bce = F.binary_cross_entropy(p, mask, reduction="none")

    return (u * bce).mean()