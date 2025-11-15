# train.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from model import CrossAttnVLM
from preprocess import make_loader, Vocab


def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 데이터 / vocab
    ds, dl = make_loader(
        json_dir=args.json_dir,
        jpg_dir=args.jpg_dir,
        vocab=None,
        build_vocab=True,
        shuffle=True,
    )

    vocab = ds.dataset.vocab if isinstance(ds, torch.utils.data.Subset) else ds.vocab
    print("Vocab size:", len(vocab.itos))

    # 2) 모델
    model = CrossAttnVLM(len(vocab.itos), dim=args.dim, pretrained_backbone=not args.no_pretrain).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    total = len(ds)
    print("Train samples:", total)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for imgs, ids, lens, targets, metas in dl:
            valid_idx = [i for i, t in enumerate(targets) if t is not None]
            if not valid_idx:
                continue

            imgs = imgs[valid_idx].to(device)
            ids = ids[valid_idx].to(device)
            lens = lens[valid_idx].to(device)
            t = torch.stack([targets[i] for i in valid_idx], dim=0).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(imgs, ids, lens)
                # (cx, cy, w, h) - y 축에 조금 더 가중치
                loss_cx = F.smooth_l1_loss(pred[:, 0], t[:, 0])
                loss_cy = F.smooth_l1_loss(pred[:, 1], t[:, 1])
                loss_w = F.smooth_l1_loss(pred[:, 2], t[:, 2])
                loss_h = F.smooth_l1_loss(pred[:, 3], t[:, 3])
                loss = (1 * loss_cx + 2 * loss_cy + 1 * loss_w + 2 * loss_h) / 6.0

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * imgs.size(0)

        scheduler.step()
        avg = running / max(1, total)
        print(f"[Epoch {ep}/{args.epochs}] loss={avg:.4f}")

    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_itos": vocab.itos,
            "dim": args.dim,
            "img_size": args.img_size,
            "no_pretrain": args.no_pretrain,
        },
        args.save_ckpt,
    )
    print("✅ Saved ckpt to", args.save_ckpt)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", default=CFG.TRAIN_JSON_DIR)
    p.add_argument("--jpg_dir", default=CFG.TRAIN_JPG_DIR)
    p.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    p.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
    p.add_argument("--dim", type=int, default=CFG.DIM)
    p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    p.add_argument("--footer_crop_ratio", type=float, default=CFG.FOOTER_CROP_RATIO)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--save_ckpt", default=CFG.CKPT_PATH)
    return p.parse_args()


if __name__ == "__main__":
    seed_everything()
    args = parse_args()
    # DataLoader 전역 설정용으로 CFG 값도 맞춰두기
    CFG.IMG_SIZE = args.img_size
    CFG.BATCH_SIZE = args.batch_size
    CFG.NUM_WORKERS = args.num_workers
    CFG.FOOTER_CROP_RATIO = args.footer_crop_ratio
    train(args)
