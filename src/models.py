import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from torchvision.models import vit_b_32, ViT_B_32_Weights


class MHA(nn.Module):
    def __init__(self, img_size, hidden_dim, patch_size, n_attention_heads, use_self_attention, in_channels=3):
        super().__init__()

        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        _num_pos = (img_size // patch_size) ** 2
        in_size = in_channels * patch_size**2
        self.positionals = nn.Parameter(
            torch.randn(1, _num_pos, in_size)
        )

        self.norm = nn.LayerNorm(in_size)
        self.use_self_attention = use_self_attention

        self.q = nn.Linear(in_size,hidden_dim)
        self.k = nn.Linear(in_size,hidden_dim)
        self.v = nn.Linear(in_size,hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_attention_heads, batch_first=True)

    def forward(self, x):
        patched, n_patches = self.patchify(x)
        patched = patched.transpose(-2,-3).flatten(-2)

        patched = patched + self.positionals
        # eventually dropout
        normalized = self.norm(patched)

        if self.use_self_attention:
            q = self.q(normalized)
            k = self.k(normalized)
            v = self.v(normalized)
        else:
            q = normalized
            k = normalized
            v = normalized
        
        x = self.attention(q,k,v)

        return x


    def patchify(self, img):
        patch_size = self.patch_size

        if img.shape[-1] % patch_size != 0:
            to_pad = img.shape[-1] - (img.shape[-1] % patch_size) # TODO: rivedere
            img = transforms.functional.pad(img, to_pad//2)
        n_patches = img.shape[-1] // patch_size

        patched = torch.zeros(*img.shape[:-2], n_patches ** 2, patch_size ** 2, device=img.device)
        for i in range(n_patches):
            for j in range(n_patches):
                start_i = i * patch_size
                end_i = (i+1) * patch_size
                start_j = j * patch_size
                end_j = (j+1) * patch_size
                patched[...,n_patches*i+j,:] = img[...,start_i:end_i, start_j:end_j].reshape(*img.shape[:-2],patch_size**2)
        
        return patched, n_patches
    
    @staticmethod
    def unpatchify(patched_img):
        # patched_img: batch_size x num_patches x patch_size
        num_patches, patch_size = patched_img.shape[-2:]
        img_size = int(num_patches ** (1/2))
        patch_size = int(patch_size ** (1/2))
        unpatched = torch.zeros(*patched_img.shape[:-2], img_size * patch_size, img_size * patch_size, device=patched_img.device)

        for i in range(img_size):
            for j in range(img_size):
                start_i = i * patch_size
                end_i = (i+1) * patch_size
                start_j = j * patch_size
                end_j = (j+1) * patch_size
                unpatched[..., start_i:end_i, start_j:end_j] = patched_img[...,i*patch_size+j,:].reshape(-1,patch_size, patch_size)
        
        return unpatched
    
class CustomBackbone(nn.Module):
    def __init__(self, in_size, use_self_attention, hidden_dim, patch_size=16, n_attention_heads=4, in_channels=3):
        super().__init__()
        self.use_self_attention = use_self_attention
        self.hidden_dim = hidden_dim

        self.mha = MHA(in_size, hidden_dim, patch_size, n_attention_heads, use_self_attention, in_channels)
        self.conv_2 = nn.Conv2d(in_channels, 7, 8, padding="same")
        
        self.conv_3 = nn.Conv2d(8, 32, 7, 2)
        self.conv_4 = nn.Conv2d(32, 72, 4, padding="same")
        self.conv_5 = nn.Conv2d(72, 196, 3)
        self.conv_6 = nn.Conv2d(196, 256, 3, padding="same")

    def forward(self, x):
        focus, focus_mask = self.mha(x)
        features = self.conv_2(x).relu()
        features = F.max_pool2d(features, 2)

        # Fusion
        focus = self.mha.unpatchify(focus).unsqueeze(1)
        fused = torch.cat([focus, features], dim=1)

        fused = self.conv_3(fused).relu()
        fused = F.max_pool2d(fused, 3)

        fused = self.conv_4(fused).relu()
        fused = self.conv_5(fused).relu()
        fused = F.max_pool2d(fused, 2)
        fused = self.conv_6(fused).relu()

        return fused

class MultiObjectNet(nn.Module):
    def __init__(self, in_size, max_objects, num_classes, backbone):
        super().__init__()

        self.name = "MultiObjectNet"
        self.in_size = in_size
        self.num_classes = num_classes
        self.max_objects = max_objects

        self.backbone = backbone

        self.queries = nn.Parameter(
            torch.randn(max_objects, 256)
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(256, 4, dim_feedforward=1024, batch_first=True),
            3,
        )

        self.category = nn.Linear(256, num_classes)
        self.boxes = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        batch_size, n_channels, img_size, _ = x.shape
        
        #fused = fused.flatten(-3,-1)
        features = self.backbone(x)
        # ALT #
        features = features.flatten(-2)
        features = features.permute(0,2,1) # batch_size, num_channels, h*w -> batch_size, h*w, num_channels
        queries = self.queries.expand(batch_size, -1,-1)
        features = self.decoder(queries, features)
        #####

        categories = self.category(features)#.reshape(batch_size, self.max_objects, self.num_classes)
        boxes = self.boxes(features).sigmoid()#.reshape(batch_size, self.max_objects, 4)

        return categories, boxes
    

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SimpleNet"
        self.net = vit_b_32(ViT_B_32_Weights.DEFAULT)
        self.net.heads = nn.Identity()
        self.net.requires_grad_(True)
        self.category = nn.Linear(768, 10*14)
        self.boxes = nn.Linear(768, 10*4)

    def forward(self,x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        batch_size, n_channels, img_size, _ = x.shape
        #x = x.reshape(batch_size, n_channels*img_size*img_size)
        x = self.net(x)
        #print(x.shape)
        category = self.category(x).reshape(batch_size, 10, 14)
        boxes = self.boxes(x).reshape(batch_size, 10, 4)
        boxes = nn.functional.sigmoid(boxes)

        return category, boxes
