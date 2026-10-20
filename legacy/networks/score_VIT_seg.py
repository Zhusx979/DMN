import torch.nn
import torchvision

# model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained =  False)
# model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=1, stride=1)
import torch.nn.functional as F
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pdb


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
        self.linear = nn.Linear(in_features, out_features, bias=False)
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

        self.dim = dim
        self.heads = heads

        self.norm = nn.LayerNorm(dim)

        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.C = ScoreLinear(self.dim, self.dim)
        self.C.register_forward_pre_hook(pre_forward_hook)

        self.W = torch.nn.Linear(self.dim, self.inner_dim)
        project_out = not (heads == 1 and dim_head == dim)

        self.to_out = (
            nn.Sequential(nn.Linear(self.inner_dim, self.dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        v = self.C(x)
        y = self.W(v)
        return self.to_out(y)

    
    def calculate_loss(self):
        c = self.C.linear.weight.data
        row_sum = c.sum(dim=1) - 1.0  # 计算每一行和减去1
        sum_loss = row_sum.pow(2).mean()  # 取平方并计算均值
        
        lower_loss = torch.mean(torch.relu(-c) ** 2) # ensure c>0
        upper_loss = torch.mean(torch.relu(c-1) ** 2) # ensure c<1
        # logger.debug("sum loss: {}, lower loss: {}, upper loss: {}".format(sum_loss, lower_loss, upper_loss))
        return  sum_loss + lower_loss + upper_loss

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

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class score_ViTSeg(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=48, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        # Patch embedding（保持不变）
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ScoreTransformer(image_size, patch_size, dim, depth, heads, dim_head, mlp_dim, dropout)
        self.feat_h = image_height // patch_height
        self.feat_w = image_width // patch_width

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.segmentation_head = nn.Conv2d(dim // 8, num_classes, kernel_size=1)

        self.final_upsample = nn.Upsample(scale_factor=patch_height, mode='bilinear', align_corners=False)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 1:, :]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.feat_h, w=self.feat_w)
        x = self.decoder(x)
        logits = self.segmentation_head(x)
        if logits.shape[2:] != img.shape[2:]:
            logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear', align_corners=False)

        return logits
    
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

if __name__ == '__main__':
    input = torch.rand((1,1,256, 256))
    model = score_ViTSeg(image_size=256, patch_size=16, num_classes=5, dim=768, depth=8, heads=16, mlp_dim=3072, channels=1)

    output_tensor = model(input)
    print(model)
    print(f"Input shape {input.shape}")
    print(f"output shape {output_tensor.shape}")