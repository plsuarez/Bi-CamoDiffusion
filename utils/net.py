# net.py  
import os
import warnings
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention

# ----------------------------
# Utils: shapes
# ----------------------------

def _squeeze_b1(x: torch.Tensor) -> torch.Tensor:
    # (B,1,C,H,W)->(B,C,H,W)
    if torch.is_tensor(x) and x.dim() == 5 and x.shape[1] == 1:
        return x.squeeze(1)
    return x

def _ensure_edge_1ch(edge: torch.Tensor) -> torch.Tensor:
    if edge is None or (not torch.is_tensor(edge)):
        return None
    edge = _squeeze_b1(edge)  # (B,1,H,W) if (B,1,1,H,W)
    if edge.dim() == 3:
        edge = edge.unsqueeze(1)
    if edge.dim() == 4 and edge.shape[1] != 1:
        edge = edge[:, :1, ...]
    return edge


# ----------------------------
# PVT blocks
# ----------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.mask_chans = int(mask_chans)
        if self.mask_chans != 0:
            self.mask_proj = nn.Conv2d(self.mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # init 
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)

        if (mask is not None) and (self.mask_chans != 0):
            # mask
            mask = self.mask_proj(mask)
            x = x + mask

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        mask_chans=1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = int(mask_chans)

        self.time_embed = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
            embed_dim=embed_dims[0], mask_chans=mask_chans
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
            embed_dim=embed_dims[1], mask_chans=0
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
            embed_dim=embed_dims[2], mask_chans=0
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
            embed_dim=embed_dims[3], mask_chans=0
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x_mask, timesteps, cond_img):
        # x_mask: (B,mask_chans,H,W) 
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0])).unsqueeze(1)

        B = cond_img.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(cond_img, x_mask)
        x = torch.cat([time_token, x], dim=1)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1])).unsqueeze(1)
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2])).unsqueeze(1)
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3])).unsqueeze(1)
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x_mask, timesteps, cond_img):
        return self.forward_features(x_mask, timesteps, cond_img)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


# ----------------------------
# PVT variants
# ----------------------------

class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs
        )


# ----------------------------
# decoder 
# ----------------------------

from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class conv(nn.Module):
    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def Downsample(dim, dim_out=None, factor=2):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None, factor=2):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2
        dim_out = dim if dim_out is None else dim_out
        conv1 = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(conv1, nn.SiLU(), nn.PixelShuffle(factor))
        self.init_conv_(conv1)

    def init_conv_(self, conv1):
        o, i, h, w = conv1.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w, device=conv1.weight.device)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)
        conv1.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv1.bias.data)

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super().__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        self.down = nn.Sequential(
            ConvModule(in_channels=mask_chans, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs, timesteps, x_mask):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        c1, c2, c3, c4 = inputs

        # down on x_mask
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x_mask = blk(x_mask, t)
            else:
                x_mask = blk(x_mask)

        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))

        x = torch.cat([_c, x_mask], dim=1)

        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)

        x = self.pred(x)
        return x, c1, c2, c3, c4


# ----------------------------
# FINAL NET (edge-aware)
# ----------------------------

class net(nn.Module):

    def __init__(self, class_num=2, mask_chans=1, **kwargs):
        super().__init__()
        self.class_num = class_num
        self.mask_chans = int(mask_chans)

        self.backbone = pvt_v2_b4_m(in_chans=3, mask_chans=self.mask_chans)
        self.decode_head = Decoder(
            dims=[64, 128, 320, 512],
            dim=256,
            class_num=class_num,
            mask_chans=self.mask_chans
        )

        # 
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, self.mask_chans, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mask_chans, self.mask_chans, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.edge_proj[-1].weight)
        nn.init.zeros_(self.edge_proj[-1].bias)

        self._init_weights()

    def extract_features(self, cond_img, edge=None, **kwargs):
        # 
        return cond_img

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, conditioning_features, edge=None, **kwargs):
        return self.forward(x, timesteps, conditioning_features, edge=edge, **kwargs)

    def forward(self, x, timesteps, conditioning_features, edge=None, **kwargs):
        """
        x: (B,mask_chans,H,W) noisy mask (+ extra_cond 
        conditioning_features: RGB (B,3,H,W)
        edge: (B,1,H,W) 
        """
        cond_img = conditioning_features
        x = _squeeze_b1(x)
        cond_img = _squeeze_b1(cond_img)

        edge = _ensure_edge_1ch(edge)
        if edge is not None:
            # match H,W
            if edge.shape[-2:] != x.shape[-2:]:
                edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)

            # edge to  [0,1]
            if edge.min().item() < 0.0:
                edge01 = (edge + 1.0) * 0.5
            else:
                edge01 = edge
            edge01 = edge01.clamp(0.0, 1.0)

            edge_feat = self.edge_proj(edge01)
            x = x + edge_feat

        features = self.backbone(x, timesteps, cond_img)
        features, *_ = self.decode_head(features, timesteps, x)
        return features

    def _download_weights(self, model_name):
        _available_weights = ['pvt_v2_b4_m']
        assert model_name in _available_weights
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        pretrained_dict = torch.load(self._download_weights('pvt_v2_b4_m'), map_location="cpu")
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass
