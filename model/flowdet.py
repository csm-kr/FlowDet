import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

import rootutils
rootutils.setup_root(__file__, indicator='.vscode', pythonpath=True)

from typing import Optional, Callable, Union
from torch import Tensor

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # dinov3 모델 로드
        self.model = torch.hub.load('./dinov3', 'dinov3_vits16', source='local', weights='dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        # 평가 모드로 설정
        # self.model.eval()  
        # # 파라미터 업데이트 방지
        # for param in self.model.parameters():
        #     param.requires_grad = False  

    def forward(self, x):

        features = self.model.forward_features(x)
        x = features['x_norm_patchtokens'] # [B, 196, 384] # B, 14*14, 384
        # x = x.view(x.size(0), 14, 14, 384)
        return x

 
# =========================================================
# Adaptive LayerNorm Zero (AdaLN-Zero)
# =========================================================
class AdaLayerNormZero(nn.Module):
    """
    AdaLN-Zero: LayerNorm with adaptive scale and bias (and gating)
    Used in e.g., Stable Diffusion, T2I-Adapter, DiT, etc.
    """
    def __init__(self, dim, time_embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 6 * dim)  # 6×dim: 3 layers × (scale, shift, gate)
        )

    def forward(self, x, t_emb):
        """
        x: (B, N, D)
        t_emb: (B, time_embed_dim)
        """
        scale1, shift1, gate1, scale2, shift2, gate2 = self.mlp(t_emb).chunk(6, dim=-1)
        x_norm = self.norm(x)
        return (scale1.unsqueeze(1) + 1) * x_norm + shift1.unsqueeze(1), (scale2, shift2, gate1, gate2)


# =========================================================
# Transformer Block with Self-Attn + Cross-Attn + FFN + AdaLN-Zero
# =========================================================
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        time_embed_dim=512,
        cross_attn=True
    ):
        super().__init__()
        self.cross_attn_enabled = cross_attn

        # Attention layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if cross_attn:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Adaptive LayerNorm Zero
        self.adaln = AdaLayerNormZero(d_model, time_embed_dim)

        self.act = nn.GELU()

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None
    ):
        """
        x: (B, N, D)
        context: (B, M, D)  (encoder memory for cross-attn)
        timestep: (B, time_embed_dim)
        """

        # === Adaptive LayerNorm Zero ===
        x_mod, (scale2, shift2, gate1, gate2) = self.adaln(x, timestep)

        # === 1. Self-Attention ===
        sa_out, _ = self.self_attn(x_mod, x_mod, x_mod,
                                   attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask,
                                   need_weights=False)
        x = x + gate1.unsqueeze(1) * self.dropout_attn(sa_out)

        # === 2. Cross-Attention (optional) ===
        if self.cross_attn_enabled and context is not None:
            ca_norm = self.adaln.norm(x)
            ca_out, _ = self.cross_attn(ca_norm, context, context, need_weights=False)
            x = x + gate1.unsqueeze(1) * self.dropout_attn(ca_out)

        # === 3. Feed-Forward ===
        x_ff_norm = self.adaln.norm(x)
        x_ff = (1 + scale2.unsqueeze(1)) * x_ff_norm + shift2.unsqueeze(1)
        x_ff = self.linear2(self.dropout(self.act(self.linear1(x_ff))))
        x = x + gate2.unsqueeze(1) * self.dropout(x_ff)

        return x


class FlowDet(nn.Module):
    def __init__(self, args=None):
        super(FlowDet, self).__init__()

        self.args = args
        self.feature_extractor = FeatureExtractor()
        if self.args is not None:
            d_model = args.d_model
            depth = args.depth
        else:
            d_model = 384
            depth = 6

        # bounding box 임베딩
        self.box_embed = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 시간 임베딩
        self.time_embed = nn.Sequential(
            Rearrange("b -> b 1"),
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.dit = nn.ModuleList([TransformerBlock(d_model=d_model, nhead=8, time_embed_dim=d_model, cross_attn=True) for _ in range(depth)])
        self.box_header = nn.Linear(d_model, 4)
        self.objectness_header = nn.Linear(d_model, 1)

    def forward(self, i, x, t):

        # get init box
        x = self.box_embed(x)

        # 시간 임베딩
        t = self.time_embed(t)
        
        # 이미지 특징 추출
        f = self.feature_extractor(i)

        # dit block
        for block in self.dit:
            x = block(x, f, timestep=t)

        pred_box = self.box_header(x)
        pred_objectness = self.objectness_header(x)
        return (pred_box, pred_objectness)
    
    def train_step(model, optimizer, sampler, images, batch_size=1, num_boxes=50, device="cuda"):
        model.train()
        optimizer.zero_grad()

        # 1. 샘플링
        x0, x1, t = sampler.sample(batch_size, num_boxes)
        xt = sampler.sample_xt(x0, x1, t)
        ut = sampler.compute_target_flow(x0, x1, t, xt)

        # 2. 모델 forward
        pred_box, pred_obj = model(images, xt, t)

        # 3. Flow Matching Loss (rectified flow)
        flow_loss = torch.mean((pred_box - ut) ** 2)

        # 4. Objectness Loss (focal-like, 간단 버전)
        target_obj = torch.ones_like(pred_obj)
        obj_loss = torch.mean((pred_obj - target_obj) ** 2)

        total_loss = flow_loss + 0.1 * obj_loss
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "obj_loss": obj_loss.item()
        }
    
    @torch.no_grad()
    def inference_euler(model, image, x0, steps=10, device="cuda"):
        """
        Euler integration을 이용한 rectified flow 추론
        """
        model.eval()
        dt = 1.0 / steps
        xt = x0.clone().to(device)
        t = torch.zeros(x0.shape[0], device=device)

        for step in range(steps):
            # 모델이 예측한 flow (velocity)
            pred_box, _ = model(image, xt, t)

            # Euler update: x_{t+Δt} = x_t + v_t * Δt
            xt = xt + pred_box * dt
            t = t + dt

        return xt  # 최종 예측 (approx x1)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.randn(1, 3, 224, 224).to(device)
    boxes = torch.randn(1, 50, 4).to(device)
    times = torch.tensor([0.5]).to(device)

    model = FlowDet().to(device)
    x1, x2 = model(image, boxes, times)
    print('the shape of x1:', x1.shape)
    print('the shape of x2:', x2.shape)

    # 전체 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())

    # 학습 가능한 파라미터만
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")