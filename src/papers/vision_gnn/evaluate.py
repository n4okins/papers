"""
https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
"""

# %%
import argparse
import json
import math
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

# import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm

torch.cuda.empty_cache()

class TypedArgs(argparse.Namespace):
    data_dir: Path
    model_path: Path
    model_type: Literal["pvig", "vig"] = "vig"
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int | None = None
    batch_size: int = 64
    k: int = 9
    conv: Literal["edge", "mr", "sage", "gin"] = "mr"
    act: Literal["relu", "gelu"] = "gelu"
    norm: Literal["batch", "instance"] = "batch"
    bias: bool = True
    n_blocks: int = 16
    n_filters: int = 320
    n_classes: int = 1000
    dropout_rate: float = 0.0
    droppath_rate: float = 0.0
    use_dilation: bool = True
    epsilon: float = 0.2
    use_stochastic: bool = False
    blocks: list[int] = [2, 2, 6, 2]
    channels: list[int] = [80, 160, 400, 640]
    emb_dims: int = 1024

    def to_model_config(self):
        if self.model_type == "vig":
            return dict(
                k=self.k,
                conv=self.conv,
                act=self.act,
                norm=self.norm,
                bias=self.bias,
                n_blocks=self.n_blocks,
                n_filters=self.n_filters,
                n_classes=self.n_classes,
                dropout_rate=self.dropout_rate,
                droppath_rate=self.droppath_rate,
                use_dilation=self.use_dilation,
                epsilon=self.epsilon,
                use_stochastic=self.use_stochastic,
            )
        elif self.model_type == "pvig":
            return dict(
                k=self.k,
                conv=self.conv,
                act=self.act,
                norm=self.norm,
                bias=self.bias,
                n_classes=self.n_classes,
                dropout_rate=self.dropout_rate,
                droppath_rate=self.droppath_rate,
                use_dilation=self.use_dilation,
                epsilon=self.epsilon,
                use_stochastic=self.use_stochastic,
                blocks=self.blocks,
                channels=self.channels,
                emb_dims=self.emb_dims,
            )

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "TypedArgs":
        return TypedArgs(**vars(args))

#region
# ================================================================
# ImageNet id to name
# ================================================================
IMAGENET_CLASSES = json.loads(
    requests.get("https://gist.githubusercontent.com/n4okins/8354136b39ed5c21d48a7ee95f790ad1/raw/5028ff689bc368440858a0fe578ad72780ee77a4/ImageNet_ja.json").text
)
IMAGENET_ID_TO_CLASSES = {n["num"]:dict(index=i, **n) for i, n in enumerate(IMAGENET_CLASSES)}

# ================================================================
# DropPath from timm.models.layers
# ================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

# ================================================================

def get_activation(act: str, **kwargs: dict):
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'silu':
        return nn.SiLU()
    else:
        return nn.Identity()

def get_normalization( norm: str, out_channels: int, **kwargs: dict):
    if norm == 'batch':
        return nn.BatchNorm2d(out_channels, affine=True)
    elif norm == 'instance':
        return nn.InstanceNorm2d(out_channels, affine=False)
    else:
        return nn.Identity()
# ================================================================
# Some funcs and modules from Vision GNN gcn_lib
# ================================================================


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            # (3, 192, 196, 1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class BasicConv(nn.Sequential):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(get_normalization(norm, out_channels=channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(get_activation(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()



# ================================================================
# Grapher, FFN, Stem, DeepGCN from Vision GNN
# ================================================================

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = get_activation(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class StemViG(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            get_activation(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            get_activation(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            get_activation(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            get_activation(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class StemPViG(nn.Module):
    """Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            get_activation(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            get_activation(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class DeepGCNViG(torch.nn.Module):
    def __init__(
        self, k: int = 9, conv: Literal["edge", "mr", "sage", "gin"] = "mr",
        act: Literal["relu", "gelu"] = "gelu", norm: Literal["batch", "instance"] = "batch",
        bias: bool = True, n_blocks: int = 16, n_filters: int = 320, n_classes: int = 1000,
        dropout_rate: float = 0.1, droppath_rate: float = 0.0, use_dilation: bool = True,
        epsilon: float = 0.2, use_stochastic: bool = False
    ):
        """
        """
        super(DeepGCNViG, self).__init__()

        self.n_blocks = n_blocks
        self.stem = StemViG(out_dim=n_filters, act=act)

        dpr = [x.item() for x in torch.linspace(0, droppath_rate, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, n_filters, 14, 14))

        if use_dilation:
            self.backbone = nn.Sequential(*[nn.Sequential(Grapher(n_filters, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, use_stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(n_filters, n_filters * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = nn.Sequential(*[nn.Sequential(Grapher(n_filters, num_knn[i], 1, conv, act, norm,
                                                bias, use_stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(n_filters, n_filters * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = nn.Sequential(nn.Conv2d(n_filters, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              get_activation(act),
                              nn.Dropout(dropout_rate),
                              nn.Conv2d(1024, n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

class Downsample(nn.Module):
    """Convolution-based downsample"""

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DeepGCNPViG(torch.nn.Module):
    def __init__(
        self,
        k: int = 9,
        conv: Literal["edge", "mr", "sage", "gin"] = "mr",
        act: Literal["relu", "gelu"] = "gelu",
        norm: Literal["batch", "instance"] = "batch",
        bias: bool = True,
        n_classes: int = 1000,
        dropout_rate: float = 0,
        droppath_rate: float = 0.1,
        use_dilation: bool = True,
        epsilon: float = 0.2,
        use_stochastic: bool = False,
        blocks: list[int] = [2, 2, 6, 2],
        channels: list[int] = [80, 160, 400, 640],
        emb_dims: int = 1024,
    ):

        super(DeepGCNPViG, self).__init__()
        self.n_blocks = sum(blocks)
        reduce_ratios = [4, 2, 1, 1]
        dpr = [
            x.item() for x in torch.linspace(0, droppath_rate, self.n_blocks)
        ]  # stochastic depth decay rule
        num_knn = [
            int(x.item()) for x in torch.linspace(k, k, self.n_blocks)
        ]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.stem = StemPViG(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    nn.Sequential(
                        Grapher(
                            channels[i],
                            num_knn[idx],
                            min(idx // 4 + 1, max_dilation),
                            conv,
                            act,
                            norm,
                            bias,
                            use_stochastic,
                            epsilon,
                            reduce_ratios[i],
                            n=HW,
                            drop_path=dpr[idx],
                            relative_pos=True,
                        ),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx]),
                    )
                ]
                idx += 1
        self.backbone = nn.Sequential(*self.backbone)

        self.prediction = nn.Sequential(
            nn.Conv2d(channels[-1], 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            get_activation(act),
            nn.Dropout(dropout_rate),
            nn.Conv2d(1024, n_classes, 1, bias=True),
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

#endregion

# %%
def is_env_notebook():
    """Determine wheather is the environment Jupyter Notebook"""
    if 'get_ipython' not in globals():
        # Python shell
        return False
    else:
        env_name = globals()["get_ipython"]().__class__.__name__
        if env_name == 'TerminalInteractiveShell':
            # IPython shell
            return False
        # Jupyter Notebook
        return True

def get_args() -> TypedArgs:
    if is_env_notebook():
        # For Jupyter Notebook
        # Please set the path to the ImageNet dataset
        imagenet_dir = "/path/to/imagenet"

        args = TypedArgs(
            data_dir=imagenet_dir,
            model_path=Path(__file__).parent / "models/vig_s_80.6.pth",
            seed=None,
        )
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("dir", metavar="data_dir", type=Path)
        parser.add_argument("--model_type", default="vig", type=str)
        parser.add_argument("--model_path", type=Path, default=Path(__file__).parent / "models/vig_s_80.6.pth")
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--seed", default=None)
        parser.add_argument("--batch_size", default=128, type=int)
        parser.add_argument("--k", default=9, type=int)
        parser.add_argument("--conv", default="mr", type=str)
        parser.add_argument("--act", default="gelu", type=str)
        parser.add_argument("--norm", default="batch", type=str)
        parser.add_argument("--bias", default=True, type=bool)
        parser.add_argument("--n_blocks", default=16, type=int)
        parser.add_argument("--n_filters", default=320, type=int)
        parser.add_argument("--n_classes", default=1000, type=int)
        parser.add_argument("--dropout_rate", default=0.1, type=float)
        parser.add_argument("--droppath_rate", default=0.0, type=float)
        parser.add_argument("--use_dilation", default=True, type=bool)
        parser.add_argument("--epsilon", default=0.2, type=float)
        parser.add_argument("--use_stochastic", default=False, type=bool)
        parser.add_argument("--blocks", default=[2, 2, 6, 2], type=list)
        parser.add_argument("--channels", default=[80, 160, 400, 640], type=list)
        parser.add_argument("--emb_dims", default=1024, type=int)
        args = TypedArgs.from_argparse(parser.parse_args())

    assert args.dir.exists(), f"Data directory {args.dir} does not exist"
    assert args.model_path.exists(), f"Model path {args.model_path} does not exist"
    return args

# %%
def to_plottable(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x.squeeze(0)
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def main():
    args = get_args()
    print(f"model type {args.model_type}")
    print(f"model path {args.model_path}")
    print(f"ImageNet directory {args.dir}")
    if args.seed:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )
    split = "val"
    pathes = tuple((args.dir / split).iterdir())
    pathes = [pathes[i:i + args.batch_size] for i in range(0, len(pathes), args.batch_size)]
    xml_dir = args.dir.parents[1] / "Annotations" / "CLS-LOC" / split
    top1_correct = 0
    top5_correct = 0
    count = 0
    if args.model_type == "vig":
        model = DeepGCNViG(**args.to_model_config())
    elif args.model_type == "pvig":
        model = DeepGCNPViG(**args.to_model_config())

    if args.model_path.exists():
        model.load_state_dict(torch.load(args.model_path))

    model.to(args.device)
    model.eval()
    pbar = tqdm(pathes)

    for image_pathes in pbar:
        metadata = []
        images = []
        for image_path in image_pathes:
            xml_path = xml_dir / image_path.with_suffix(".xml").name
            xml = ET.parse(
                xml_path,
                parser=ET.XMLParser(encoding="utf-8"),
            ).getroot()
            metadata.append(IMAGENET_ID_TO_CLASSES[xml.find("object").find("name").text])
            images.append(transform(Image.open(image_path).convert("RGB")))
        
        images = torch.stack(images)
        labels = torch.tensor([m["index"] for m in metadata])
        images = images.to(args.device)

        with torch.inference_mode():
            pred_prob = model(images)

        pred_top5 = torch.topk(pred_prob.cpu(), 5)
        top1_correct += (pred_top5.indices[:, 0] == labels).sum().item()
        top5_correct += sum(label in top5 for label, top5 in zip(labels, pred_top5.indices))
        count += len(labels)
        top1_accuracy = top1_correct / count
        top5_accuracy = top5_correct / count

        pbar.set_description(f"Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}")
    
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}, Top-5 Accuracy: {top5_accuracy:.4f}")

main()
# %%