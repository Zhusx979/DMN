import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from loguru import logger
import pdb


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ScoreLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ScoreLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias = False)
        # self.linear.weight.data = torch.rand(out_features, in_features) 
    
    def forward(self, x):
        return self.linear(x)
    

        
def pre_forward_hook(module, inputs):
     if isinstance(module, ScoreLinear):
            with torch.no_grad():
                # logger.info("invoke pre_forward_hook")
                # module.linear.weight.data = module.linear.weight.data / module.linear.weight.data.sum(dim=1, keepdim=True)
                module.linear.weight.data = torch.softmax(module.linear.weight.data, dim=1)
        

class ScoreAttention(nn.Module):
    def __init__(self, image_size, patch_size, dim, heads, dim_head, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
        self.dim=dim
        self.heads=heads
        
        self.norm = nn.LayerNorm(dim)
        
        self.dim_head=dim_head
        self.inner_dim = dim_head * heads
        self.C=ScoreLinear((self.image_size // self.patch_size) **2+1, (self.image_size // self.patch_size) **2+1)
        self.C.register_forward_pre_hook(pre_forward_hook)
        
        self.W=torch.nn.Linear(self.dim,self.inner_dim)
        project_out = not (heads == 1 and dim_head == dim)
        
        self.to_out = (
            nn.Sequential(nn.Linear(self.inner_dim, self.dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.dropout=torch.nn.Dropout(dropout)
        

    def forward(self, x):
        x = self.norm(x)
        
        v=self.W(x)
        v=rearrange(v, "b n (h d) -> b h d n", h=self.heads)
        
        y=self.C(v)
        out=rearrange(y, "b h d n-> b n (h d)")
        return self.to_out(out)
    

class ScoreTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ScoreAttention(image_size, patch_size, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ScoreViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=32,
        patch_size=16,
        num_classes=100,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ScoreTransformer(image_size, patch_size, dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)