import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# --- 설정 (Configuration) ---
class CFG:
    # Multilingual BERT 임베딩 크기
    EMBEDDING_DIM = 768
    # ResNet50 Layer 4 output dim
    VISUAL_FEATURE_DIM = 2048 
    # GRU의 은닉 상태 크기
    GRU_HIDDEN_DIM = 512
    # BBox 좌표 개수: [x, y, w, h] (normalized)
    BBOX_DIM = 4 

# --- Vision Encoder (ResNet50) ---
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.output_dim = CFG.VISUAL_FEATURE_DIM
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

# --- Language Encoder (Bi-GRU) ---
class LanguageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # LLM/Transformer 백본 사용 회피를 위해 Bi-GRU 사용
        self.gru = nn.GRU(
            input_size=CFG.EMBEDDING_DIM,
            hidden_size=CFG.GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.output_dim = 2 * CFG.GRU_HIDDEN_DIM
        
    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(embeddings)
        return output

# --- Cross-Attention Fusion ---
class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        D_FUSION = 2 * CFG.GRU_HIDDEN_DIM 
        
        self.query_proj = nn.Linear(D_FUSION, D_FUSION)
        self.kv_proj = nn.Linear(CFG.VISUAL_FEATURE_DIM, D_FUSION)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=D_FUSION,
            num_heads=8,
            batch_first=True
        )
        self.final_proj = nn.Sequential(
            nn.Linear(D_FUSION, D_FUSION),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, visual_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        
        # 1. 시각 특징 평탄화 및 변환
        visual_features_flat = visual_features.flatten(2).transpose(1, 2) 
        
        # 2. 특징 프로젝션
        Q = self.query_proj(language_features)  
        KV = self.kv_proj(visual_features_flat)  
        
        # 3. 크로스 어텐션 
        fused_output, _ = self.attn(
            query=Q, 
            key=KV, 
            value=KV
        ) 
        
        # 4. 최종 융합 벡터 (언어 시퀀스 평균 풀링)
        fused_vector = fused_output.mean(dim=1) 
        return self.final_proj(fused_vector)

# --- BBox Predictor ---
class BBoxPredictor(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, CFG.BBOX_DIM), 
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# --- 최종 Vision-Language Model ---
class VisionLanguageModel(nn.Module):
    def __init__(self, embeddings_layer: nn.Module = None):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.language_encoder = LanguageEncoder()
        self.fusion_module = CrossAttentionFusion()
        self.predictor = BBoxPredictor(in_features=self.language_encoder.output_dim) 
        
        # 임베딩 레이어는 외부에서 받음
        self.embeddings = embeddings_layer

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        
        # 1. 이미지 인코딩
        visual_features = self.vision_encoder(images)
        
        # 2. 언어 임베딩 (Multilingual BERT 임베딩 사용)
        if self.embeddings is None:
            raise ValueError("Embeddings layer is not attached.")
        embeddings = self.embeddings(input_ids)

        # 3. 언어 인코딩 (Bi-GRU 사용)
        language_features = self.language_encoder(embeddings, attention_mask)
        
        # 4. 특징 융합 (Cross-Attention)
        fused_vector = self.fusion_module(visual_features, language_features)
        
        # 5. BBox 예측 (0~1 정규화된 좌표)
        predicted_bbox = self.predictor(fused_vector)
        
        return predicted_bbox