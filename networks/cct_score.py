import torch
from torch import nn, einsum
import torch.nn.functional as F
import pdb
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# CCT Models

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = default(stride, max(1, (kernel_size // 2) - 1))
    padding = default(padding, max(1, (kernel_size // 2)))

    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)

# positional

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

# modules


class ScoreLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ScoreLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = False)
        # self.linear.weight.data = torch.rand(out_features, in_features) 
    
    def forward(self, x):
        return self.linear(x)
    

class ScoreAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        
        self.image_size = 224
        self.patch_size = 16
        self.dim = dim
        self.heads = num_heads
        
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5
        
        self.C=ScoreLinear((self.image_size // self.patch_size) **2, (self.image_size // self.patch_size) **2)
        
        self.W=torch.nn.Linear(self.dim,self.dim)

        # self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        v=self.W(x)
        v=rearrange(v, "b n (h d) -> b h d n", h=self.heads)
        
        y=self.C(v)
        y=rearrange(y, "b h d n-> b n (h d)")
        
        return self.proj_drop(self.proj(y))
        
    def calculate_loss(self):
        c = self.C.linear.weight.data
        row_sum = c.sum(dim=1) - 1.0  # 计算每一行和减去1
        sum_loss = row_sum.pow(2).mean()  # 取平方并计算均值
        
        lower_loss = torch.mean(torch.relu(-c) ** 2) # ensure c>0
        upper_loss = torch.mean(torch.relu(c-1) ** 2) # ensure c<1
        # logger.debug("sum loss: {}, lower loss: {}, upper loss: {}".format(sum_loss, lower_loss, upper_loss))
        return sum_loss + lower_loss + upper_loss
    
    
#     def forward(self, x):
#         B, N, C = x.shape

#         qkv = self.qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         q = q * self.scale

#         attn = einsum('b h i d, b h j d -> b h i j', q, k)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = einsum('b h i j, b h j d -> b h i d', attn, v)
#         x = rearrange(x, 'b h n d -> b n (h d)')

#         return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = ScoreAttention(
                                        dim=d_model, 
                                        num_heads=nhead,
                                   attention_dropout=attention_dropout, 
                                        projection_dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        
        attn_loss = self.self_attn.calculate_loss()
        return src, attn_loss

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(chan_in, chan_out,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return rearrange(self.conv_layers(x), 'b c h w -> b (h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                               requires_grad=True)
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        else:
            self.positional_emb = nn.Parameter(sinusoidal_embedding(sequence_length, embedding_dim),
                                               requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        b = x.shape[0]
        
        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_token, x), dim=1)

        if exists(self.positional_emb):
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x, attn_loss = blk(x)

        x = self.norm(x)

        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)
        else:
            x = x[:, 0]

        return self.fc(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# CCT Main model

class CCTScore(nn.Module):
    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        *args, **kwargs
    ):
        super().__init__()
        img_height, img_width = pair(img_size)

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)
    
    def cal_attn_loss(self, module, prefix=''):
        total_attn_loss = 0
        with torch.no_grad():
            for name, child in module.named_children():
                new_prefix = f'{prefix}.{name}' if prefix else name
                if isinstance(child, ScoreLinear):
                    # print(f"Layer {new_prefix} is constrained with softmax weights.")

                    c = module.C.linear.weight.data
                    row_sum = c.sum(dim=1) - 1.0  # 计算每一行和减去1
                    sum_loss = row_sum.pow(2).mean()  # 取平方并计算均值

                    lower_loss = torch.mean(torch.relu(-c) ** 2) # ensure c>0
                    upper_loss = torch.mean(torch.relu(c-1) ** 2) # ensure c<1

                    attn_loss = sum_loss + lower_loss + upper_loss
                    # print(sum_loss, lower_loss, upper_loss, attn_loss)

                    total_attn_loss += attn_loss
                else:
                    total_attn_loss += self.cal_attn_loss(child, new_prefix)
        return total_attn_loss

    def forward(self, x):
        x = self.tokenizer(x)
        out = self.classifier(x)
        return out
