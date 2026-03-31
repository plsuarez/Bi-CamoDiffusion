# coding=utf-8
import inspect
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
import torch.nn.functional as F


# -------------------------
# Helpers
# -------------------------

def _get_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _squeeze_if_5d(x: torch.Tensor) -> torch.Tensor:
  
    if torch.is_tensor(x) and x.dim() == 5 and x.size(1) == 1:
        return x.squeeze(1)
    return x


def _ensure_4d(x: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
   
    if x.dim() == 3:
        return x.unsqueeze(channel_dim)
    return x


def _to_float(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        x = x.float()
    return x


def _maybe_to_device(x, device: torch.device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_maybe_to_device(t, device) for t in x)
    return x


def _filter_kwargs_by_signature(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs

     
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _safe_call(fn, *args, **kwargs):
   
    fkwargs = _filter_kwargs_by_signature(fn, kwargs)
    return fn(*args, **fkwargs)


def _pred_to_01(pred: torch.Tensor) -> torch.Tensor:
    pred = _to_float(pred)

     
    pmin = float(pred.amin().detach().cpu())
    pmax = float(pred.amax().detach().cpu())

    if pmin >= -0.05 and pmax <= 1.05:
        return pred.clamp(0.0, 1.0)

   
    if pmin >= -1.2 and pmax <= 1.2:
        pred = (pred + 1.0) / 2.0
        return pred.clamp(0.0, 1.0)

    
    pred = torch.sigmoid(pred)
    return pred.clamp(0.0, 1.0)


# -------------------------
# Train / Val forward 
# -------------------------

def simple_train_val_forward(
    model,
    image: torch.Tensor,
    gt: Optional[torch.Tensor] = None,
    edge: Optional[torch.Tensor] = None,
    extra_cond: Optional[torch.Tensor] = None,
    verbose: bool = False,
    **kwargs
):
 
    return modification_train_val_forward(
        model=model,
        image=image,
        gt=gt,
        edge=edge,
        extra_cond=extra_cond,
        verbose=verbose,
        time_ensemble=False,
        gt_sizes=None,
        **kwargs
    )


# -------------------------
# Train / Val forward  
# -------------------------

@torch.no_grad()
def _sample_with_optional_edge(
    model,
    image: torch.Tensor,
    *,
    edge: Optional[torch.Tensor] = None,
    extra_cond: Optional[torch.Tensor] = None,
    verbose: bool = False,
    **kwargs
) -> torch.Tensor:
  
     
    if not hasattr(model, "sample"):
        raise AttributeError("Your model has not mode sample().")

    sample_fn = model.sample

     
    call_kwargs = dict(kwargs)
    call_kwargs.update({
        "extra_cond": extra_cond,
        "edge": edge,
        "verbose": verbose,
    })

     
    pred = _safe_call(sample_fn, image, **call_kwargs)

    # 
    if torch.is_tensor(pred):
        pred = _squeeze_if_5d(pred)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        return pred

    #  
    if isinstance(pred, (list, tuple)) and len(pred) > 0 and torch.is_tensor(pred[0]):
        pred = torch.stack([p if p.dim() == 4 else p.unsqueeze(0) for p in pred], dim=0)
        return pred

    raise TypeError("model.sample() unknown type (no tensor).")


def modification_train_val_forward(
    model,
    image: torch.Tensor,
    gt: Optional[torch.Tensor] = None,
    name: Optional[List[str]] = None,
    image_for_post: Optional[torch.Tensor] = None,
    edge: Optional[torch.Tensor] = None,
    extra_cond: Optional[torch.Tensor] = None,
    time_ensemble: bool = False,
    gt_sizes: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = False,
    **kwargs
):
 
    device = _get_device(model)

    #  
    image = _squeeze_if_5d(image)
    gt = _squeeze_if_5d(gt) if gt is not None else None
    edge = _squeeze_if_5d(edge) if edge is not None else None
    extra_cond = _squeeze_if_5d(extra_cond) if extra_cond is not None else None

    #  
    image = _maybe_to_device(image, device)
    gt = _maybe_to_device(gt, device) if gt is not None else None
    edge = _maybe_to_device(edge, device) if edge is not None else None
    extra_cond = _maybe_to_device(extra_cond, device) if extra_cond is not None else None

    #  
    if torch.is_tensor(image):
        image = _ensure_4d(image)  # (B,C,H,W)
    if gt is not None and torch.is_tensor(gt):
        gt = _ensure_4d(gt)        # (B,1,H,W)  
    if edge is not None and torch.is_tensor(edge):
        edge = _ensure_4d(edge)    # (B,1,H,W)  

    # -------------------------
    # TRAIN:  
    # -------------------------
    if model.training:
        if gt is None:
            raise ValueError("In training need 'gt'.")

        forward_fn = model.forward

        call_kwargs = dict(kwargs)
        call_kwargs.update({
            "img": gt,
            "cond_img": image,
            "seg": gt,
            "extra_cond": extra_cond,
            "edge": edge,
        })

         
        loss = _safe_call(forward_fn, **call_kwargs)

         
        if not torch.is_tensor(loss):
            raise TypeError("No tensor loss by foward.")
        return loss

    # -------------------------
    # EVAL:  
    # -------------------------
    pred = _sample_with_optional_edge(
        model,
        image,
        edge=edge,                 
        extra_cond=extra_cond,
        verbose=verbose,
        **kwargs
    )

    pred = _pred_to_01(pred)   

    out: Dict[str, Any] = {}

    if time_ensemble:
        if gt_sizes is None:
            raise ValueError("time_ensemble=True needs gt_sizes=[(H,W), ...].")

         
        preds_list = []
        b = pred.shape[0]
        for i in range(b):
            hi, wi = gt_sizes[i]
            pi = F.interpolate(pred[i:i+1], size=(hi, wi), mode="bilinear", align_corners=False)
            preds_list.append(pi.squeeze(0))   
        out["pred"] = preds_list
    else:
        out["pred"] = pred   

     
    out["image"] = image_for_post if image_for_post is not None else image
    if gt is not None:
        out["gt"] = gt
    if edge is not None:
        out["edge"] = edge

    return out
