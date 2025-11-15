import os
import json
import random
from glob import glob
from PIL import Image
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertModel, BertTokenizer 

# --- 설정 (Configuration) ---
class CFG:
    SEED = 42
    IMG_SIZE = 512
    MAX_QUERY_LEN = 128
    # KoBERT 대신 설치가 용이한 Multilingual BERT 체크포인트 사용
    TOKENIZER_PATH = 'bert-base-multilingual-cased'  

# --- 유틸리티 함수 ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_bbox(bbox: List[float], width: float, height: float) -> List[float]:
    """BBox [x, y, w, h]를 0~1 사이의 정규화된 좌표로 변환"""
    x, y, w, h = bbox
    x_norm = x / width
    y_norm = y / height
    w_norm = w / width
    h_norm = h / height
    return [x_norm, y_norm, w_norm, h_norm]

def denormalize_bbox(bbox_norm: List[float], width: float, height: float) -> List[float]:
    """정규화된 BBox를 원본 해상도 기준의 절대 좌표로 역변환"""
    x_norm, y_norm, w_norm, h_norm = bbox_norm
    x_abs = x_norm * width
    y_abs = y_norm * height
    w_abs = w_norm * width
    h_abs = h_norm * height
    return [x_abs, y_abs, w_abs, h_abs]

# --- Dataset 클래스 ---
class CustomDataset(Dataset):
    def __init__(self, root_dir: str, mode: str, transform=None, tokenizer=None):
        self.mode = mode
        self.transform = transform
        # BertTokenizer 사용
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(CFG.TOKENIZER_PATH)
        self.data_list = self._load_data(root_dir, mode)

    def _load_data(self, root_dir: str, mode: str) -> List[Dict[str, Any]]:
        """제시된 폴더 구조에 맞게 데이터 경로를 설정하고 파일을 로드합니다."""
        data_list = []
        
        data_folder_name = 'train_data' if mode in ['train', 'val'] else 'test_data'
        data_sub_folder_name = mode if mode in ['train', 'val'] else 'test'
        
        base_path = os.path.join(root_dir, data_folder_name, data_sub_folder_name) 

        if mode in ['train', 'val']:
            for doc_type in ['press', 'report']:
                json_path = os.path.join(base_path, f'{doc_type}_json')
                
                # --- ✅ 수정된 부분: 이미지 폴더 이름을 press/report 모두 '_jpg'로 통일 ---
                img_folder_name = f'{doc_type}_jpg'
                img_path = os.path.join(base_path, img_folder_name)

                for json_file in glob(os.path.join(json_path, '*.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        continue 
                    
                    img_filename = data['source_data_info']['source_data_name_jpg']
                    img_file = os.path.join(img_path, img_filename)
                    resolution = data['source_data_info']['document_resolution'] # [W, H]

                    for anno in data['learning_data_info']['annotation']:
                        class_name = anno.get('class_name', '')
                        if class_name in ['표', '차트(세로 막대형)', '차트(꺾은선형)', '차트'] and 'visual_instruction' in anno:
                            data_list.append({
                                'img_path': img_file,
                                'query': anno['visual_instruction'],
                                'bbox': anno['bounding_box'],  # [x, y, w, h]
                                'instance_id': anno['instance_id'],
                                'resolution': resolution
                            })

        elif mode == 'test':
            query_path = os.path.join(base_path, 'query') 
            img_path = os.path.join(base_path, 'images') 

            for json_file in glob(os.path.join(query_path, '*.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                
                    img_filename = data['source_data_info']['source_data_name_jpg']
                    img_file = os.path.join(img_path, img_filename)
                    resolution = data['source_data_info']['document_resolution'] # [W, H]

                    for anno in data['learning_data_info']['annotation']:
                        if 'visual_instruction' in anno:
                            data_list.append({
                                'img_path': img_file,
                                'query': anno['visual_instruction'],
                                'instance_id': anno['instance_id'],
                                'resolution': resolution
                            })
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        data = self.data_list[idx]
        
        # 1. 이미지 로드 및 변환
        image = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        # 2. 질의 토큰화
        tokenized_query = self.tokenizer(
            data['query'],
            padding='max_length',
            truncation=True,
            max_length=CFG.MAX_QUERY_LEN,
            return_tensors='pt'
        )
        
        # 3. 레이블(BBox) 처리
        width, height = data['resolution']
        
        if self.mode in ['train', 'val']:
            # 훈련/검증: BBox 정규화
            bbox_norm = normalize_bbox(data['bbox'], width, height)
            bbox_tensor = torch.tensor(bbox_norm, dtype=torch.float)
            
            return image_tensor, tokenized_query['input_ids'].squeeze(0), tokenized_query['attention_mask'].squeeze(0), bbox_tensor

        elif self.mode == 'test':
            # 테스트: Instance ID와 해상도 반환
            return image_tensor, tokenized_query['input_ids'].squeeze(0), tokenized_query['attention_mask'].squeeze(0), data['instance_id'], torch.tensor([width, height], dtype=torch.float)


# --- 데이터셋/로더 생성 함수 ---
def get_data_loaders(root_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained(CFG.TOKENIZER_PATH)
    
    train_dataset = CustomDataset(root_dir, 'train', transform=transform, tokenizer=tokenizer)
    val_dataset = CustomDataset(root_dir, 'val', transform=transform, tokenizer=tokenizer)
    test_dataset = CustomDataset(root_dir, 'test', transform=transform, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader