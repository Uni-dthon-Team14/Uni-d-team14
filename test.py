import os
import json
import argparse
import pandas as pd
import torch
from glob import glob
from torch.utils.data import DataLoader

# ✔ model.py에서 모델만 가져옴
from model import CrossAttnVLM

# ✔ Vocab, UniDSet는 preprocess.py에서 가져옴
from preprocess import Vocab, UniDSet, collate_fn


# ------------------------
# 모델 로드
# ------------------------
def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    vocab = Vocab()
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}

    # img_size 제거하여 baseline과 호환
    model = CrossAttnVLM(
        vocab_size=len(vocab.itos),
        dim=ckpt["dim"],
        pretrained_backbone=not ckpt.get("no_pretrain", False)
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, vocab


# ------------------------
# 예측 + 최종 제출 생성
# ------------------------
def run_predict_and_filter(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_ckpt(args.ckpt, device)

    # JSON 파일 로드
    json_files = sorted(glob(os.path.join(args.json_dir, "*.json")))

    # 테스트셋 로더
    ds = UniDSet(json_files, jpg_dir=args.jpg_dir, vocab=vocab, build_vocab=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2, collate_fn=collate_fn)

    rows = []
    with torch.no_grad():
        for imgs, ids, lens, targets, meta in dl:
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)

            pred = model(imgs, ids, lens)

            for i in range(len(meta)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = pred[i].cpu().numpy().tolist()

                x = (cx - nw / 2) * W
                y = (cy - nh / 2) * H
                w = nw * W
                h = nh * H

                rows.append({
                    "query_id": meta[i]["query_id"],
                    "query_text": meta[i]["query_text"],
                    "pred_x": x,
                    "pred_y": y,
                    "pred_w": w,
                    "pred_h": h
                })

    pred_df = pd.DataFrame(rows)
    print("예측 결과 총 개수:", len(pred_df))

    # --------------------------
    # sample_submission 기반 필터링
    # --------------------------
    sample_df = pd.read_csv(args.sample_csv)
    valid_ids = set(sample_df["query_id"])

    filtered = pred_df[pred_df["query_id"].isin(valid_ids)]
    print("필터링 후 개수:", len(filtered))

    # 순서 맞추기
    filtered = filtered.set_index("query_id").loc[sample_df["query_id"]].reset_index()

    os.makedirs(os.path.dirname(args.final_csv), exist_ok=True)
    filtered.to_csv(args.final_csv, index=False, encoding="utf-8-sig")
    print(f"[완료] 최종 제출 파일 저장 → {args.final_csv}")


# ------------------------
# main
# ------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--jpg_dir", required=True)
    parser.add_argument("--sample_csv", default="./open/sample_submission.csv")
    parser.add_argument("--final_csv", default="outputs/preds/final_submit.csv")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    run_predict_and_filter(args)


if __name__ == "__main__":
    main()
