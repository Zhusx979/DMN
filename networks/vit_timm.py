import torch
import torch.nn as nn
import timm

class ViT_timm(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.extractor = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        self.extractor.patch_embed.proj = nn.Conv2d(4, 768, kernel_size=(16, 16), stride=(16, 16))
        self.extractor.head = nn.Linear(768, hidden_dim)
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)
    
    def forward(self, x):
        x = self.extractor(x)
        z = self.classifier(x)
        return z