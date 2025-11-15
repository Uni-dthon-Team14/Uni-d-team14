# preprocess.py
import os
import json
import glob
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision import transforms as T
    _BACKBONE_OK = True
except:
    _BACKBONE_OK = False
    T = None

from config import CFG


############################################################
# JSON 검색
############################################################
def find_jsons(json_dir: str):
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"json_dir not found → {json_dir}")
    return sorted(glob.glob(os.path.join(json_dir, "*.json")))


############################################################
# JSON 로딩
############################################################
def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


############################################################
# 이미지 매칭
############################################################
def get_image_path(json_path, data, jpg_dir):
    candidates = []
    info = data.get("source_data_info", {})

    # 우선순위 후보들
    for key in ["source_data_name_jpg", "source_image", "source_data_name"]:
        if key in info:
            val = str(info[key]).strip()
            if val:
                candidates.append(val)

    # JSON베이스 이름 기반 이미지
    base = os.path.splitext(os.path.basename(json_path))[0]
    candidates += [f"{base}.jpg", f"{base}.png"]

    candidates = list(set(candidates))
    if jpg_dir is None:
        return None

    all_imgs = os.listdir(jpg_dir)

    # 1) 정확히 일치
    for c in candidates:
        p = os.path.join(jpg_dir, c)
        if os.path.exists(p):
            return p

    # 2) 부분 일치
    for c in candidates:
        key = c.replace(".jpg", "").replace(".png", "")
        for f in all_imgs:
            if key in f:
                return os.path.join(jpg_dir, f)

    print(f"[MISS IMG] {json_path} | candidates={candidates}")
    return None


############################################################
# 토큰화
############################################################
def simple_tokenize(s: str):
    if not s:
        return []
    for ch in [",", "(", ")", ":", "?", "!", "·"]:
        s = s.replace(ch, " ")
    return s.strip().split()


############################################################
# Vocab
############################################################
class Vocab:
    def __init__(self):
        self.freq = {}
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {tok:i for i,tok in enumerate(self.itos)}

    def build(self, texts):
        for t in texts:
            for tok in simple_tokenize(t):
                self.freq[tok] = self.freq.get(tok, 0) + 1

        for tok, _ in sorted(self.freq.items(), key=lambda x: -x[1]):
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, s, max_len=40):
        toks = simple_tokenize(s)[:max_len]
        if not toks:
            return [1]
        return [self.stoi.get(t, 1) for t in toks]


############################################################
# Dataset (✨ JSON 1개 = row 1개 보장)
############################################################
class UniDSet(Dataset):
    def __init__(self, json_files, jpg_dir, vocab, build_vocab=False,
                 resize_to=(CFG.IMG_SIZE, CFG.IMG_SIZE),
                 test_mode=False):

        self.items = []
        for jf in json_files:
            data = read_json(jf)
            ann_list = data.get("learning_data_info", {}).get("annotation", [])
            img_path = get_image_path(jf, data, jpg_dir)

            # --------------------------
            # ⭐ 핵심 : JSON 1개당 row 1개 생성
            # --------------------------

            if len(ann_list) == 0:
                # annotation 없음 → dummy row
                self.items.append({
                    "json": jf,
                    "img": img_path,
                    "query": "",
                    "bbox": None,
                    "query_id": os.path.basename(jf).replace(".json", "")
                })

            else:
                # annotation 있음 → 첫 번째 것만 사용
                a = ann_list[0]
                self.items.append({
                    "json": jf,
                    "img": img_path,
                    "query": a.get("visual_instruction", ""),
                    "bbox": a.get("bounding_box"),
                    "query_id": a.get("instance_id"),
                })

        # vocab
        self.vocab = vocab
        if build_vocab:
            self.vocab.build([it["query"] for it in self.items])

        # transforms
        self.resize_to = resize_to
        if _BACKBONE_OK:
            from torchvision import transforms as T
            self.tf = T.Compose([T.Resize(resize_to), T.ToTensor()])
        else:
            self.tf = None

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _pil_to_tensor(img):
        arr = np.array(img).astype(np.float32) / 255.
        if len(arr.shape)==2:
            arr = np.stack([arr]*3, -1)
        arr = arr.transpose(2,0,1)
        return torch.tensor(arr)

    def __getitem__(self, idx):
        it = self.items[idx]

        if it["img"] is None:
            img = Image.new("RGB", self.resize_to, (0,0,0))
            W0, H0 = self.resize_to
        else:
            img = Image.open(it["img"]).convert("RGB")
            W0, H0 = img.size

        # resize
        if self.tf:
            img_t = self.tf(img)
        else:
            img = img.resize(self.resize_to)
            img_t = self._pil_to_tensor(img)

        # tokenize
        ids = self.vocab.encode(it["query"])
        L = len(ids)

        return {
            "image": img_t,
            "query_ids": torch.tensor(ids),
            "length": torch.tensor(L),
            "target": None,                # test mode no bbox
            "orig_size": (W0, H0),
            "query_id": it["query_id"],
            "query_text": it["query"]
        }


############################################################
# Collate
############################################################
def collate_fn(batch):
    B = len(batch)
    max_len = max(b["length"] for b in batch)

    ids = torch.zeros(B, max_len, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    imgs = torch.stack([b["image"] for b in batch])

    metas = []
    for i,b in enumerate(batch):
        L = b["length"]
        ids[i,:L] = b["query_ids"][:L]
        lens[i] = L
        metas.append({
            "orig_size": b["orig_size"],
            "query_id": b["query_id"],
            "query_text": b["query_text"]
        })

    return imgs, ids, lens, [None]*B, metas


############################################################
# Loader
############################################################
def make_loader(json_dir, jpg_dir, vocab, build_vocab,
                batch_size=CFG.BATCH_SIZE, img_size=CFG.IMG_SIZE,
                num_workers=CFG.NUM_WORKERS, shuffle=False):

    json_files = find_jsons(json_dir)

    ds = UniDSet(
        json_files=json_files,
        jpg_dir=jpg_dir,
        vocab=vocab,
        build_vocab=build_vocab,
        resize_to=(img_size,img_size),
        test_mode=True
    )

    dl = DataLoader(
        ds, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_fn
    )

    return ds, dl
