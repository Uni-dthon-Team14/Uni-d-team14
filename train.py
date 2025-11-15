import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
# Multilingual BERT 모델에서 임베딩 레이어 추출을 위해 BertModel 임포트
from transformers import BertModel 

from preprocess import get_data_loaders, seed_everything, CFG
from model import VisionLanguageModel

# --- 손실 함수 (Loss Function) ---
class BBoxLoss(nn.Module):
    """BBox 예측을 위한 L1 Loss"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(pred_bbox, gt_bbox)

# --- 학습 루프 ---
def train_loop(args: argparse.Namespace):
    seed_everything(CFG.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Multilingual BERT 모델에서 임베딩 레이어만 로드
    print(f"Loading BERT embeddings layer from {CFG.TOKENIZER_PATH}...")
    # 전체 모델을 로드하되, 언어 인코더로는 사용하지 않고, 임베딩 레이어만 추출합니다.
    bert_model = BertModel.from_pretrained(CFG.TOKENIZER_PATH)
    # word_embeddings만 추출하여 트랜스포머 백본 사용을 회피합니다.
    embedding_layer = bert_model.embeddings.word_embeddings.to(device)
    
    # 2. 데이터 로더
    print(f"Loading data from {args.data_root}...")
    train_loader, val_loader, _ = get_data_loaders(args.data_root, args.batch_size, args.num_workers)

    # 3. 모델 정의
    model = VisionLanguageModel(embeddings_layer=embedding_layer).to(device)
    
    # 4. 옵티마이저 및 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = BBoxLoss()

    best_val_loss = float('inf')
    
    print(f"Start training on {device} for {args.epochs} epochs.")
    
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)")
        for images, input_ids, attention_mask, gt_bbox in pbar:
            images, input_ids, attention_mask, gt_bbox = images.to(device), input_ids.to(device), attention_mask.to(device), gt_bbox.to(device)
            
            optimizer.zero_grad()
            
            predicted_bbox = model(images, input_ids, attention_mask)
            
            loss = criterion(predicted_bbox, gt_bbox)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, input_ids, attention_mask, gt_bbox in val_loader:
                images, input_ids, attention_mask, gt_bbox = images.to(device), input_ids.to(device), attention_mask.to(device), gt_bbox.to(device)
                
                predicted_bbox = model(images, input_ids, attention_mask)
                loss = criterion(predicted_bbox, gt_bbox)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- 모델 저장 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, f'best_model.pth')
            torch.save(model.state_dict(), ckpt_path) 
            print(f"Saved best model checkpoint to {ckpt_path}")

def main():
    parser = argparse.ArgumentParser(description="Vision-Language Model Training")
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory of the dataset (default: ./data)')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    args = parser.parse_args()
    train_loop(args)

if __name__ == '__main__':
    main()