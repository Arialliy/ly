# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Zicheng Feng
# @File    : MTTU_Net.py
# @Software: PyCharm
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import time
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile


def get_MTTU_Net_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.QKV_size = 128 * 4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.transformer_channel = 128
    config.n_classes = 1

    # ********** useless **********
    # config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        x_3,x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)
        return x


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Channel-wise Cross Attention
class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class Normalized_Patch_Embedding(nn.Module):
    def __init__(self, patchsize, in_channels, out_channels):
        super().__init__()
        patch_size = _pair(patchsize)

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        # self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class D_ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(D_ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv_fig = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + 2 * self.hidden_dim, out_channels=3 * self.hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=3 * self.hidden_dim, out_channels=3 * self.hidden_dim, kernel_size=3, padding=1, bias=False, groups=3 * self.hidden_dim)
        )

        self.conv_o1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1, bias=False, groups=self.hidden_dim)
        )

        self.conv_o2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1, bias=False, groups=self.hidden_dim)
        )

    def forward(self, input_tensor, cur_state):
        h1_cur, h2_cur, c_cur = cur_state

        fig = self.conv_fig(torch.cat([input_tensor, h1_cur, h2_cur], dim=1))
        cc_f, cc_i, cc_g = torch.split(fig, self.hidden_dim, dim=1)

        cc_o1 = self.conv_o1(torch.cat([input_tensor, h1_cur], dim=1))
        cc_o2 = self.conv_o2(torch.cat([input_tensor, h2_cur], dim=1))

        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        g = torch.tanh(cc_g)
        o1 = torch.sigmoid(cc_o1)
        o2 = torch.sigmoid(cc_o2)

        c_next = f * c_cur + i * g
        h1_next = o1 * torch.tanh(c_next)
        h2_next = o2 * torch.tanh(c_next)

        return h1_next, h2_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_fig[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_fig[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_fig[0].weight.device))


class PPModule(nn.Module):
    # (1, 3, 6, 8)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


# TSTM Cross Attention (TSTM CA)
class Attention_TSTM(nn.Module):
    def __init__(self, config, vis):
        super(Attention_TSTM, self).__init__()
        self.vis = vis
        self.QKV_size = config.transformer_channel
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        self.mheadq = nn.Conv2d(self.QKV_size, self.QKV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q = nn.Conv2d(self.QKV_size * self.num_attention_heads, self.QKV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.QKV_size * self.num_attention_heads, bias=False)

        self.project_out = nn.Conv2d(self.QKV_size, self.QKV_size, kernel_size=1, bias=False)

        self.pp = PPModule((1, 3, 6, 8))

        self.convlstm = D_ConvLSTMCell(self.QKV_size, self.QKV_size)

    def forward(self, emb_all, prev_state):
        b, c, h, w = emb_all.shape

        q = self.q(self.mheadq(emb_all))

        if prev_state == None:
            prev_state = self.convlstm.init_hidden(b, [h, w])

        cur_state = self.convlstm(emb_all, prev_state)
        k, v = cur_state[0], cur_state[1]

        k = self.pp(k)
        v = self.pp(v)

        q = rearrange(q, '(b n) (head c) h w -> b head (n h w) c ', head=self.num_attention_heads, n=4)
        k = rearrange(k, '(b n) (head c) l -> b head (n l) c ', head=self.num_attention_heads, n=4)
        v = rearrange(v, '(b n) (head c) l -> b head (n l) c ', head=self.num_attention_heads, n=4)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.QKV_size)

        attention_probs = self.softmax(self.psi(attn))

        out = (attention_probs @ v)

        out = out.mean(dim=1)

        out = rearrange(out, 'b (n h w) c -> (b n) c h w', h=h, w=w, n=4)

        Out = self.project_out(out)

        weights = attn

        return Out, cur_state, weights


#  Temporal Space Transformer Module (TSTM)
class TSTM(nn.Module):
    def __init__(self, config, vis):
        super(TSTM, self).__init__()
        self.attn_norm = LayerNorm3d(config.transformer_channel, LayerNorm_type='WithBias')

        self.channel_attn = Attention_TSTM(config, vis)

        self.ffn_norm = LayerNorm3d(config.transformer_channel, LayerNorm_type='WithBias')

        self.ffn = FeedForward(config.transformer_channel, ffn_expansion_factor=2.66, bias=False)

    def forward(self, emb1, emb2, emb3, emb4, prev_state):
        embcat = []
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=0)

        org = emb_all
        cx_all = self.attn_norm(emb_all)
        cx_all, cur_state, weights = self.channel_attn(cx_all, prev_state)
        cx_all = org + cx_all

        org = cx_all

        x_all = self.ffn_norm(cx_all)

        x_all = self.ffn(x_all)

        x_all = x_all + org

        x1, x2, x3, x4 = x_all.chunk(4, dim=0)

        weights = x1
        return x1, x2, x3, x4, cur_state, weights


# TCTM Cross Attention (TCTM CA)
class Attention_TCTM(nn.Module):
    def __init__(self, config, vis):
        super(Attention_TCTM, self).__init__()
        self.vis = vis
        self.QKV_size = config.QKV_size
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        self.mheadq = nn.Conv2d(self.QKV_size, self.QKV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q = nn.Conv2d(self.QKV_size * self.num_attention_heads, self.QKV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.QKV_size * self.num_attention_heads, bias=False)

        self.project_out = nn.Conv2d(self.QKV_size, self.QKV_size, kernel_size=1, bias=False)

        self.convlstm = D_ConvLSTMCell(config.QKV_size, config.QKV_size)

    def forward(self, emb_all, prev_state):
        b, c, h, w = emb_all.shape
        q = self.q(self.mheadq(emb_all))

        if prev_state == None:
            prev_state = self.convlstm.init_hidden(b, [h, w])

        cur_state = self.convlstm(emb_all, prev_state)
        k, v = cur_state[0], cur_state[1]

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.QKV_size)

        attention_probs = self.softmax(self.psi(attn))

        out = (attention_probs @ v)

        out = out.mean(dim=1)

        out = rearrange(out, 'b  c (h w) -> b c h w', h=h, w=w)

        Out = self.project_out(out)

        weights = attn

        return Out, cur_state, weights


#  Temporal Channel Transformer Module (TCTM)
class TCTM(nn.Module):
    def __init__(self, config, vis):
        super(TCTM, self).__init__()
        self.attn_norm = LayerNorm3d(config.QKV_size, LayerNorm_type='WithBias')

        self.channel_attn = Attention_TCTM(config, vis)

        self.ffn_norm = LayerNorm3d(config.QKV_size, LayerNorm_type='WithBias')

        self.ffn = FeedForward(config.QKV_size, ffn_expansion_factor=2.66, bias=False)

    def forward(self, emb1, emb2, emb3, emb4, prev_state):
        embcat = []
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)

        org = emb_all
        cx_all = self.attn_norm(emb_all)
        cx_all, cur_state, weights = self.channel_attn(cx_all, prev_state)
        cx_all = org + cx_all

        org = cx_all

        x_all = self.ffn_norm(cx_all)

        x_all = self.ffn(x_all)

        x_all = x_all + org

        x1, x2, x3, x4 = x_all.chunk(4, dim=1)

        weights = x1

        return x1, x2, x3, x4, cur_state, weights

#  Memory-based Temporal Transformer Module (MTTM)
class MTTM(nn.Module):
    def __init__(self, config, vis):
        super(MTTM, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(config.transformer_channel, LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(config.transformer_channel, LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(config.transformer_channel, LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(config.transformer_channel, LayerNorm_type='WithBias')
        for i in range(config.transformer["num_layers"]): # 4
            if (i % 2) == 0:
                layer = TCTM(config, vis)
            else:
                layer = TSTM(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4, prev_memory_list):
        attn_weights = []
        cur_memory_list = []
        l = 0
        attn_weights.append(emb1)
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, cur_state, weights = layer_block(emb1, emb2, emb3, emb4, prev_memory_list[l])
            cur_memory_list.append(cur_state)
            l += 1
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)
        emb4 = self.encoder_norm4(emb4)
        return emb1, emb2, emb3, emb4, cur_memory_list, attn_weights


class MTTM_Process(nn.Module):
    def __init__(self, config, vis, channel_num=[32, 64, 128, 256], patchSize=[16, 8, 4, 2]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Normalized_Patch_Embedding(self.patchSize_1, in_channels=channel_num[0], out_channels=config.transformer_channel)
        self.embeddings_2 = Normalized_Patch_Embedding(self.patchSize_2, in_channels=channel_num[1], out_channels=config.transformer_channel)
        self.embeddings_3 = Normalized_Patch_Embedding(self.patchSize_3, in_channels=channel_num[2], out_channels=config.transformer_channel)
        self.embeddings_4 = Normalized_Patch_Embedding(self.patchSize_4, in_channels=channel_num[3], out_channels=config.transformer_channel)
        self.mttm = MTTM(config, vis)

        self.reconstruct_1 = Reconstruct(config.transformer_channel, channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(config.transformer_channel, channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(config.transformer_channel, channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(config.transformer_channel, channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4, prev_memory_list):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, cur_memory_list, attn_weights = self.mttm(emb1, emb2, emb3, emb4, prev_memory_list)

        x1 = self.reconstruct_1(encoded1)
        x2 = self.reconstruct_2(encoded2)
        x3 = self.reconstruct_3(encoded3)
        x4 = self.reconstruct_4(encoded4)

        x1 = x1 + en1
        x2 = x2 + en2
        x3 = x3 + en3
        x4 = x4 + en4

        return x1, x2, x3, x4, cur_memory_list, attn_weights


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension

        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

# Memory-based Temporal U-Net (MTTU-Net)
class MTTU_Net(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_layers = config.transformer.num_layers
        in_channels = config.base_channel  # basic channel 32
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.mttm_p = MTTM_Process(config, vis,
                                channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                patchSize=config.patch_sizes)
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x, prev_memory_list=None):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))

        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4

        if prev_memory_list == None:
            prev_memory_list = [None] * self.num_layers
        x1, x2, x3, x4, cur_memory_list, att_weights = self.mttm_p(x1, x2, x3, x4, prev_memory_list)

        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4

        d4 = self.up_decoder4(d5, x4)
        d3 = self.up_decoder3(d4, x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))
        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)

            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out)), cur_memory_list
            else:
                return torch.sigmoid(out), cur_memory_list
        else:
            return torch.sigmoid(out), cur_memory_list


if __name__ == '__main__':
    device = 'cuda:0'
    config_vit = get_MTTU_Net_config()
    model = MTTU_Net(config_vit, mode='test', deepsuper=True)
    model = model.to(device)
    model.eval()
    inputs = torch.rand(1, 1, 512, 640).to(device)
    cur_memory_list = None
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            output, cur_memory_list = model(inputs, cur_memory_list)
    end_time = time.time()
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('Out Size = ', output[-1].shape)
    print('Time = ', (end_time-start_time)/100)
    print('FPS = ',  100 / (end_time - start_time))
    print('FLOPs = ' + str(2 * flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
