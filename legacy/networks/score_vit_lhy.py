import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from loguru import logger


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class ScoreLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ScoreLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # self.linear.weight.data = torch.rand(out_features, in_features) 
    
    def forward(self, x):
        return self.linear(x)
    

# def backward_hook(module, grad_input, grad_output):
#     if isinstance(module, ScoreLinear):
        # module.linear.weight.data = module.linear.weight.data.clamp(min=0) 
        # module.linear.weight.data = module.linear.weight.data / module.linear.weight.data.sum(dim=1, keepdim=True) 
        # module.linear.weight.data = torch.softmax(module.linear.weight.data, dim=1)


# def modify_inputs(module, inputs):
#     modified_inputs = (inputs[0] * 2,)  # Multiply inputs by 2
#     return modified_inputs
        
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
        self.dim_head=dim_head
        self.inner_dim = dim_head * heads
        self.C=ScoreLinear((self.image_size // self.patch_size) **2+1, (self.image_size // self.patch_size) **2+1)
        # self.C.register_backward_hook(backward_hook)
        self.C.register_forward_pre_hook(pre_forward_hook)
        self.W=torch.nn.Linear(self.dim,self.inner_dim)
        project_out = not (heads == 1 and dim_head == dim)
        self.to_out = (
            nn.Sequential(nn.Linear(self.inner_dim, self.dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.dropout=torch.nn.Dropout(dropout)
        

#     def calculate_loss(self):
#         c = self.C.linear.weight.data
#         row_sum = c.sum(dim=1) - 1.0  # 计算每一行和减去1
#         sum_loss = row_sum.pow(2).mean()  # 取平方并计算均值
        
#         lower_loss = torch.mean(torch.relu(-c) ** 2) # ensure c>0
#         upper_loss = torch.mean(torch.relu(c-1) ** 2) # ensure c<1
#         logger.info("sum loss: {}, lower loss: {}, upper loss: {}".format(sum_loss, lower_loss, upper_loss))
#         return sum_loss + lower_loss + upper_loss
    

    # def enforce_constraint(self):
    #     with torch.no_grad():
    #         self.C.linear.weight.data = F.relu(self.C.linear.weight.data)  # 使用ReLU函数处理权重矩阵

    def forward(self, x):
        v=self.W(x)
        v=rearrange(v, "b n (h d) -> b h d n", h=self.heads)
        y=self.C(v)
        out=rearrange(y, "b h d n-> b n (h d)")
        return self.to_out(out)

# class Scoreweight(nn.Module):
#     def __init__(self, depth):
#         super(Scoreweight, self).__init__()
#         self.attention_weights = nn.Parameter(torch.full((depth,), 0.1))


# def weight_hook(module, grad_input, grad_output):
#     if isinstance(module, Scoreweight):
#         module.attention_weights.data = torch.clamp(module.attention_weights.data, min=0)
#         module.attention_weights.data /= module.attention_weights.data.sum()



class ScoreTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # self.at=Scoreweight(depth)
        # self.at.register_backward_hook(weight_hook)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            ScoreAttention(
                                image_size = image_size, patch_size = patch_size, dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )
    
    def forward(self, x):
        # total_loss = 0.0
        for i, (attn, ff) in enumerate(self.layers):

            # attn_module = attn.fn  # 获取注意力模块
            x_attn = attn(x) + x  # 使用注意力模块进行计算
            
            # attn_loss = attn_module.calculate_loss()  # 计算注意力权重矩阵的损失
            # attn_loss_weighted = attn_loss * self.at.attention_weights[i]  # 加权计算损失
            # total_loss += 0.1*attn_loss

            x_ff = ff(x_attn) + x_attn  # 使用第一个元素进行前馈网络计算
            x = x_ff + x

        return x


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
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
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

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


# net=ModViT()
# hook_handle = net.register_backward_hook(constraint_hook)