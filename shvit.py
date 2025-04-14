import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, SqueezeExcite, trunc_normal_


class LayerNormChannel(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        x = weight * x + bias
        return x


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bn_weight_init=1.0,
        act_layer=nn.ReLU,
        bias=False,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=bias,
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        if bn_weight_init is not None:
            nn.init.constant_(self.bn.weight, bn_weight_init)
            nn.init.constant_(self.bn.bias, 0)
        if act_layer is not None:
            self.add_module("act", act_layer())

    @torch.no_grad()
    def fuse(self):
        modules = list(self._modules.values())
        if (
            len(modules) == 2
            and isinstance(modules[0], nn.Conv2d)
            and isinstance(modules[1], nn.BatchNorm2d)
        ):
            conv, bn = modules[0], modules[1]
            w_bn = bn.weight / (bn.running_var + bn.eps) ** 0.5
            w_conv = conv.weight * w_bn.view(-1, 1, 1, 1)
            b_conv = bn.bias - bn.running_mean * w_bn

            fused_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
                device=conv.weight.device,
            )
            fused_conv.weight.data.copy_(w_conv)
            fused_conv.bias.data.copy_(b_conv)
            return fused_conv
        elif (
            len(modules) == 3
            and isinstance(modules[0], nn.Conv2d)
            and isinstance(modules[1], nn.BatchNorm2d)
            and isinstance(modules[2], nn.Identity)
        ):
            fused_conv = self.fuse()
            return nn.Sequential(fused_conv, modules[2])
        else:
            return self


class NormLinear(nn.Sequential):
    def __init__(self, in_features, out_features, bias=True, std=0.02):
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(in_features))
        self.add_module("linear", nn.Linear(in_features, out_features, bias=bias))
        trunc_normal_(self.linear.weight, std=std)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        norm, linear = self._modules.values()
        mu = norm.running_mean
        var = norm.running_var
        gamma = norm.weight
        beta = norm.bias
        eps = norm.eps
        W = linear.weight
        b_linear = linear.bias

        w_bn = gamma / (var + eps).sqrt()
        b_bn = beta - mu * w_bn

        W_fused = W * w_bn[None, :]

        b_fused = (
            (b_bn @ W.T)
            if b_linear is None
            else ((W @ b_bn[:, None]).view(-1) + b_linear)
        )

        fused_linear = nn.Linear(
            linear.in_features, linear.out_features, device=W.device
        )
        fused_linear.weight.data.copy_(W_fused)
        fused_linear.bias.data.copy_(b_fused)
        return fused_linear


# --- Core Building Blocks ---


class ResidualWithDrop(nn.Module):
    def __init__(self, m: nn.Module, drop_prob=0.0):
        super().__init__()
        self.m = m
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.m(x))

    @torch.no_grad()
    def fuse(self):
        if hasattr(self.m, "fuse"):
            fused_m = self.m.fuse()
            if (
                isinstance(fused_m, nn.Conv2d)
                and fused_m.groups == fused_m.in_channels
                and fused_m.kernel_size == (3, 3)
            ):
                identity = torch.ones(
                    fused_m.weight.shape[0],
                    fused_m.weight.shape[1],
                    1,
                    1,
                    device=fused_m.weight.device,
                )
                identity = F.pad(identity, [1, 1, 1, 1])
                fused_m.weight.data += identity
                return fused_m
            else:
                self.m = fused_m
                return self
        else:
            return self


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bn_weight_init=0.0):
        super().__init__()
        self.pw1 = ConvBnAct(embed_dim, hidden_dim, act_layer=nn.ReLU)
        self.pw2 = ConvBnAct(
            hidden_dim, embed_dim, act_layer=None, bn_weight_init=bn_weight_init
        )

    def forward(self, x):
        return self.pw2(self.pw1(x))


class SHSA(nn.Module):
    def __init__(
        self, dim, qk_dim, partial_dim, pruning_ratio=0.0
    ):  # Added pruning_ratio
        super().__init__()
        assert 0.0 <= pruning_ratio < 1.0, "pruning_ratio must be in [0, 1)"
        self.scale = qk_dim**-0.5
        self.qk_dim = qk_dim
        self.v_dim = partial_dim
        self.partial_dim = partial_dim
        self.dim = dim
        self.pruning_ratio = pruning_ratio
        self.num_tokens_to_keep = -1

        self.pre_norm = LayerNormChannel(partial_dim)
        self.qkv = ConvBnAct(partial_dim, qk_dim * 2 + self.v_dim, act_layer=None)
        self.proj = nn.Sequential(
            nn.ReLU(), ConvBnAct(dim, dim, act_layer=None, bn_weight_init=0.0)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        assert C == self.dim, f"Input channel {C} != expected dim {self.dim}"

        x_attn, x_skip = torch.split(
            x, [self.partial_dim, self.dim - self.partial_dim], dim=1
        )
        x_norm = self.pre_norm(x_attn)

        qkv = self.qkv(x_norm)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.v_dim], dim=1)

        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)

        attn_logits = (q.transpose(-2, -1) @ k) * self.scale
        attn_weights = attn_logits.softmax(dim=-1)

        if self.pruning_ratio > 0.0:
            importance_scores = attn_weights.sum(dim=1)
            N_keep = max(1, int(N * (1.0 - self.pruning_ratio)))

            keep_indices = torch.topk(
                importance_scores, k=N_keep, dim=-1, largest=True
            ).indices
            keep_indices = torch.sort(keep_indices, dim=-1)[0]

            indices_expanded_v = keep_indices.unsqueeze(1).expand(-1, self.v_dim, -1)
            v_pruned = torch.gather(v, 2, indices_expanded_v)

            indices_expanded_attn = keep_indices.unsqueeze(1).expand(-1, N, -1)
            attn_weights_pruned_cols = torch.gather(
                attn_weights.transpose(-1, -2), 2, indices_expanded_attn
            )

            attn_output = v_pruned @ attn_weights_pruned_cols.transpose(-2, -1)

        else:
            attn_output = v @ attn_weights.transpose(-2, -1)

        attn_output = attn_output.reshape(B, self.v_dim, H, W)
        x_merged = torch.cat([attn_output, x_skip], dim=1)
        x_out = self.proj(x_merged)

        return x_out


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hid_dim = in_dim * 4
        self.conv1 = ConvBnAct(in_dim, hid_dim, kernel_size=1, act_layer=nn.ReLU)
        self.conv2_dw = ConvBnAct(
            hid_dim,
            hid_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=hid_dim,
            act_layer=nn.ReLU,
        )
        self.se = SqueezeExcite(hid_dim, rd_ratio=0.25)
        self.conv3 = ConvBnAct(hid_dim, out_dim, kernel_size=1, act_layer=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


# --- Structural Blocks ---


class BasicBlock(nn.Module):
    def __init__(
        self, dim, qk_dim, partial_dim, block_type="s", drop_prob=0.0, pruning_ratio=0.0
    ):  # Added pruning_ratio
        super().__init__()
        self.conv = ResidualWithDrop(
            ConvBnAct(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                act_layer=None,
                bn_weight_init=0.0,
            ),
            drop_prob=drop_prob,
        )

        if block_type == "s":
            self.mixer = ResidualWithDrop(
                SHSA(
                    dim, qk_dim, partial_dim, pruning_ratio=pruning_ratio
                ),  # Pass pruning_ratio
                drop_prob=drop_prob,
            )
        elif block_type == "i":
            self.mixer = nn.Identity()
        else:
            raise ValueError(f"Unknown BasicBlock type: {block_type}")

        self.ffn = ResidualWithDrop(
            FFN(dim, int(dim * 2), bn_weight_init=0.0), drop_prob=drop_prob
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, in_chans, embed_dims):
        super().__init__(
            ConvBnAct(
                in_chans,
                embed_dims[0] // 8,
                kernel_size=3,
                stride=2,
                padding=1,
                act_layer=nn.ReLU,
            ),
            ConvBnAct(
                embed_dims[0] // 8,
                embed_dims[0] // 4,
                kernel_size=3,
                stride=2,
                padding=1,
                act_layer=nn.ReLU,
            ),
            ConvBnAct(
                embed_dims[0] // 4,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                act_layer=nn.ReLU,
            ),
            ConvBnAct(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                act_layer=nn.ReLU,
            ),
        )


class DownsampleBlock(nn.Sequential):
    def __init__(self, in_dim, out_dim, drop_prob=0.0):
        super().__init__(
            ResidualWithDrop(
                ConvBnAct(
                    in_dim,
                    in_dim,
                    kernel_size=3,
                    padding=1,
                    groups=in_dim,
                    act_layer=None,
                    bn_weight_init=0.0,
                ),
                drop_prob=drop_prob,
            ),
            ResidualWithDrop(
                FFN(in_dim, int(in_dim * 2), bn_weight_init=0.0), drop_prob=drop_prob
            ),
            PatchMerging(in_dim, out_dim),
            ResidualWithDrop(
                ConvBnAct(
                    out_dim,
                    out_dim,
                    kernel_size=3,
                    padding=1,
                    groups=out_dim,
                    act_layer=None,
                    bn_weight_init=0.0,
                ),
                drop_prob=drop_prob,
            ),
            ResidualWithDrop(
                FFN(out_dim, int(out_dim * 2), bn_weight_init=0.0), drop_prob=drop_prob
            ),
        )


class SHViT(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        embed_dim=[128, 256, 384],
        partial_dim=[32, 64, 96],
        qk_dim=[16, 16, 16],
        depth=[1, 2, 3],
        types=["s", "s", "s"],
        pruning_ratios=[0.0, 0.0, 0.0],  # Added pruning ratios per stage
        distillation=False,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(embed_dim)
        self.distillation = distillation
        assert (
            len(pruning_ratios) == self.num_stages
        ), "Need one pruning ratio per stage"

        self.stem = Stem(in_chans, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.stages = nn.ModuleList()
        current_depth_idx = 0
        curr_dim = embed_dim[0]

        for i in range(self.num_stages):
            stage_depth = depth[i]
            stage_pruning_ratio = pruning_ratios[i]
            stage_blocks = []
            for j in range(stage_depth):
                drop_prob = dpr[current_depth_idx + j]
                stage_blocks.append(
                    BasicBlock(
                        dim=curr_dim,
                        qk_dim=qk_dim[i],
                        partial_dim=partial_dim[i],
                        block_type=types[i],
                        drop_prob=drop_prob,
                        pruning_ratio=stage_pruning_ratio,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            current_depth_idx += stage_depth

            if i < self.num_stages - 1:
                next_dim = embed_dim[i + 1]
                self.stages.append(
                    DownsampleBlock(
                        curr_dim, next_dim, drop_prob=dpr[current_depth_idx - 1]
                    )
                )
                curr_dim = next_dim

        self.norm = nn.BatchNorm1d(curr_dim)
        self.head = (
            nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        if distillation:
            self.head_dist = (
                nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()
            )
            trunc_normal_(self.head_dist.weight, std=0.02)
            if self.head_dist.bias is not None:
                nn.init.constant_(self.head_dist.bias, 0)

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.norm(x)

        if self.distillation:
            x_main = self.head(x)
            x_dist = self.head_dist(x)
            if self.training:
                return x_main, x_dist
            else:
                return (x_main + x_dist) / 2
        else:
            x = self.head(x)
            return x

    @torch.no_grad()
    def fuse(self):
        print(
            "Warning: Fusion with token pruning might not be fully supported or effective."
        )
        return self
        # self.stem = self._fuse_sequential(self.stem)
        # for i in range(len(self.stages)):
        #      self.stages[i] = self._fuse_sequential(self.stages[i])
        #      for block in self.stages[i]:
        #          if hasattr(block, 'fuse'):
        #              block.fuse()
        # return self

    def _fuse_sequential(self, seq):
        fused_modules = []
        for m in seq:
            # if hasattr(m, 'fuse'):
            #      fused_modules.append(m.fuse())
            # else:
            fused_modules.append(m)
        return nn.Sequential(*fused_modules)
