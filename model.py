# model.py
import math
import torch
import torch.nn as nn

from config import CFG

try:
    from torchvision.models import resnet18, ResNet18_Weights
    _TV_OK = True
except Exception:
    _TV_OK = False


# -----------------------------------------------
# Text Encoder (BiGRU)
# -----------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = CFG.DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, ids: torch.Tensor, lens: torch.Tensor):
        x = self.emb(ids)  # (B,L,D)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B,2D)
        return self.proj(h_cat)  # (B,D)


# -----------------------------------------------
# Image Encoder (ResNet18 or TinyCNN)
# -----------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, out_dim=CFG.DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=CFG.DIM, pretrained: bool = True):
        super().__init__()
        if _TV_OK:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                m = resnet18(weights=weights)
                self.backbone = nn.Sequential(*list(m.children())[:-2])  # (B,512,H/32,W/32)
                self.proj = nn.Conv2d(512, out_dim, 1)
            except Exception:
                self.backbone = TinyCNN(out_dim)
                self.proj = nn.Identity()
        else:
            self.backbone = TinyCNN(out_dim)
            self.proj = nn.Identity()

    def forward(self, x):
        f = self.backbone(x)
        return self.proj(f)  # (B,D,H',W')


# -----------------------------------------------
# Cross-Attention + BBox Head
# -----------------------------------------------
class CrossAttentionBBox(nn.Module):
    def __init__(self, dim=CFG.DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 4),
        )

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor):
        B, D, H, W = fmap.shape
        q = self.q_proj(q_vec).unsqueeze(1)  # (B,1,D)
        K = self.k_proj(fmap).flatten(2).transpose(1, 2)  # (B,HW,D)
        V = self.v_proj(fmap).flatten(2).transpose(1, 2)

        attn = torch.matmul(q, K.transpose(1, 2)) / math.sqrt(D)  # (B,1,HW)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, V).squeeze(1)  # (B,D)

        out = self.head(ctx)
        return torch.sigmoid(out)  # normalized bbox (cx,cy,w,h) in [0,1]


# -----------------------------------------------
# Final Model
# -----------------------------------------------
class CrossAttnVLM(nn.Module):
    def __init__(self, vocab_size: int, dim=CFG.DIM, pretrained_backbone=True):
        super().__init__()
        self.txt = TextEncoder(vocab_size, dim)
        self.img = ImageEncoder(dim, pretrained_backbone)
        self.head = CrossAttentionBBox(dim)

    def forward(self, imgs, ids, lens):
        q = self.txt(ids, lens)
        f = self.img(imgs)
        return self.head(q, f)


# -----------------------------------------------
# 단독 실행시: 모델 구조/파라미터 수 출력
# -----------------------------------------------
if __name__ == "__main__":
    dummy_vocab = 1000
    model = CrossAttnVLM(dummy_vocab)
    n_params = sum(p.numel() for p in model.parameters())
    print("✅ model.py loaded. #params:", n_params)
