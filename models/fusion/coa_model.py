import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

import pdb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3,
                              padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expand_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features*ffn_expand_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expand_factor=1.,
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expand_factor=ffn_expand_factor,)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3,
                      groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# =============================================================================
##########################################################################
# Layer Norm


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
        return x / torch.sqrt(sigma+1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expand_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expand_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
# Multi-DConv Head Transposed Co-Attention (MDTA)
class CoAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CoAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.query = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                     nn.Conv2d(
                                         dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                     ])
        self.key = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                   nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                             padding=1, groups=dim, bias=bias)
                                   ])
        self.value = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                     nn.Conv2d(
                                         dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                     ])
        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(
        #     dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_feat, y_feat):
        b, c, h, w = x_feat.shape

        q = self.query(x_feat)
        k = self.key(y_feat)
        v = self.value(y_feat)

        # qkv = self.qkv_dwconv(self.qkv(x_feat))
        # q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expand_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expand_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
class CoAttTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expand_factor, bias, LayerNorm_type):
        super(CoAttTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.coatt = CoAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.att   = Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expand_factor, bias)

    def forward(self, x, y):
        x = x + self.coatt(self.norm1(x), self.norm1(y))
        x = x + self.att(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


class CrossModalityBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expand_factor, bias, LayerNorm_type):
        super(CrossModalityBlock, self).__init__()

        self.vis_coattn = CoAttTransformerBlock(dim=dim, num_heads=num_heads, ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)
        self.ir_coattn  = CoAttTransformerBlock(dim=dim, num_heads=num_heads, ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, input_feat):
        vis_feat, ir_feat = input_feat[0], input_feat[1]
        vis_feat_mid = self.vis_coattn(vis_feat, ir_feat)
        ir_feat_mid  = self.ir_coattn(ir_feat, vis_feat)
        return [vis_feat_mid, ir_feat_mid]

##########################################################################
# Overlapped image patch embedding with 3x3 Conv


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class RestormerEncoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[2, 2, 2],
                 heads=[8, 8, 8],
                 ffn_expand_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(RestormerEncoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.shared_enc = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])
        
        self.vis_att_layer = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])]) 
        self.ir_att_layer  = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.coatt_layer = nn.Sequential(*[CrossModalityBlock(dim=dim, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])]) 
        self.base_layer   = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detail_layer = DetailFeatureExtraction()
        

    def forward(self, vis_img, ir_img):
        vis_patch = self.patch_embed(vis_img)
        ir_patch  = self.patch_embed(ir_img)

        vis_feat_shared = self.shared_enc(vis_patch)
        ir_feat_shared  = self.shared_enc(ir_patch)

        #pdb.set_trace()
        vis_att_feat = self.vis_att_layer(vis_feat_shared)
        ir_att_feat  = self.ir_att_layer(ir_feat_shared)
        vis_coatt_feat, ir_coatt_feat = self.coatt_layer([vis_att_feat, ir_att_feat])
        vis_base_feat = self.base_layer(vis_coatt_feat)
        ir_base_feat  = self.base_layer(ir_coatt_feat)
        vis_detail_feat = self.detail_layer(vis_coatt_feat)
        ir_detail_feat  = self.detail_layer(ir_coatt_feat)
        return vis_base_feat, vis_detail_feat, ir_base_feat, ir_detail_feat

class RestormerDecoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expand_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(RestormerDecoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.enc_l2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expand_factor=ffn_expand_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
                nn.Conv2d(int(dim), int(dim)//2, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.LeakyReLU(),
                nn.Conv2d(int(dim)//2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.enc_l2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


class RestormerAutoEncoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[2, 2, 2],
                 heads=[8, 8, 8],
                 ffn_expand_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(RestormerAutoEncoder, self).__init__()

        self.encoder = RestormerEncoder(inp_channels, out_channels, dim, num_blocks, heads, 
                                        ffn_expand_factor, bias, LayerNorm_type)
        self.decoder = RestormerDecoder(inp_channels, out_channels, dim, num_blocks, heads, 
                                        ffn_expand_factor, bias, LayerNorm_type)

        self.base_fuse = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detail_fuse = DetailFeatureExtraction()
    
    def forward(self, vis_img, ir_img):
        vis_base_feat, vis_detail_feat, ir_base_feat, ir_detail_feat = self.encoder(vis_img, ir_img) 
        fuse_base_feat   = self.base_fuse(vis_base_feat + ir_base_feat)
        fuse_detail_feat = self.detail_fuse(vis_detail_feat + ir_detail_feat)
        fuse_img, _    = self.decoder(vis_img, fuse_base_feat, fuse_detail_feat)

        if self.training:
            rec_vis_img, _ = self.decoder(vis_img, vis_base_feat, vis_detail_feat)
            rec_ir_img, _  = self.decoder(ir_img, ir_base_feat, ir_detail_feat)
            return rec_vis_img, vis_base_feat, vis_detail_feat, \
                    rec_ir_img, ir_base_feat, ir_detail_feat, \
                    fuse_img
        else:
            return fuse_img


if __name__ == '__main__':
    h, w = 128, 128
    vis_img, ir_img = torch.randn(1, 3, h, w), torch.randn(1, 3, h, w)

    restormer_ae_model = RestormerAutoEncoder()

    if torch.cuda.is_available():
        vis_img, ir_img = vis_img.cuda(), ir_img.cuda()
        restormer_ae_model = restormer_ae_model.cuda()

    restormer_ae_model.train()
    import time
    start_time = time.time()
    for i in range(30):
        rec_vis_img, vis_base_feat, vis_detail_feat, \
                rec_ir_img, ir_base_feat, ir_detail_feat, fuse_img = \
                restormer_ae_model(vis_img, ir_img)
        print(f"time: {time.time()-start_time}")
        start_time = time.time()
    import pdb
    pdb.set_trace()
        