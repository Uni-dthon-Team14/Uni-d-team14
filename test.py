import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from transformers import BertModel

from preprocess import get_data_loaders, seed_everything, denormalize_bbox, CFG
from model import VisionLanguageModel

# --- 추론 루프 ---
def predict_loop(args: argparse.Namespace):
    seed_everything(CFG.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Multilingual BERT 모델에서 임베딩 레이어만 로드
    print(f"Loading BERT embeddings layer from {CFG.TOKENIZER_PATH}...")
    bert_model = BertModel.from_pretrained(CFG.TOKENIZER_PATH)
    embedding_layer = bert_model.embeddings.word_embeddings.to(device)

    # 2. 데이터 로더
    print(f"Loading test data from {args.data_root}...")
    _, _, test_loader = get_data_loaders(args.data_root, args.batch_size, args.num_workers)

    # 3. 모델 정의 및 가중치 로드
    model = VisionLanguageModel(embeddings_layer=embedding_layer).to(device)
    
    # 체크포인트 로드
    try:
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
        print(f"Successfully loaded checkpoint from {args.ckpt_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.ckpt_path}. Please check the path and run 'train.py' first.")
        return

    # 4. 추론 실행
    model.eval()
    results = [] 
    
    print("Start prediction on test set...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Prediction")
        for images, input_ids, attention_mask, instance_ids, resolutions in pbar:
            images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
            
            # Forward Pass: 정규화된 BBox [0, 1] 예측
            predicted_bbox_norm = model(images, input_ids, attention_mask) 
            
            # 예측된 BBox를 CPU로 이동
            predicted_bbox_norm = predicted_bbox_norm.cpu().numpy()
            
            # --- BBox 역정규화 (Denormalization) ---
            for i in range(images.size(0)):
                bbox_norm = predicted_bbox_norm[i]
                instance_id = instance_ids[i]
                width, height = resolutions[i].tolist() 
                
                # 역정규화 (절대 픽셀값: x, y, w, h)
                bbox_abs = denormalize_bbox(bbox_norm, width, height) 
                
                # 결과 저장 (제출 양식에 맞춤)
                results.append({
                    'instance_id': instance_id,
                    'x': bbox_abs[0],
                    'y': bbox_abs[1],
                    'w': bbox_abs[2],
                    'h': bbox_abs[3],
                })

    # 5. 제출 파일 생성
    df_submission = pd.DataFrame(results)
    
    # 제출 파일 형식: instance_id | x | y | w | h 순서로 정렬
    df_submission = df_submission[['instance_id', 'x', 'y', 'w', 'h']]
    
    df_submission.to_csv(args.out_csv, index=False)
    print(f"\nPrediction completed and submission file saved to {args.out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Vision-Language Model Prediction")
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory of the dataset (default: ./data)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--out_csv', type=str, default='./submission.csv', help='Output path for the submission CSV file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for prediction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    args = parser.parse_args()
    predict_loop(args)

if __name__ == '__main__':
    main()