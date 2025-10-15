# trainer.py
import torch
import torch.nn.functional as F


# =========================================================
#  Rectified Flow 기반 학습 (train_step)
# =========================================================
# def train_step(model, optimizer, sampler, images, gt_boxes, batch_size=1, num_boxes=50, device="cuda"):
#     """
#     한 스텝 학습 수행 (Rectified Flow 기반)

#     Args:
#         model: FlowDet
#         optimizer: torch.optim
#         sampler: DistributionSampler
#         images: Tensor [B, 3, H, W]
#         gt_boxes: Tensor [B, N_gt, 4]
#         batch_size: int
#         num_boxes: int (샘플링할 box 수, e.g., 50)
#         device: str
#     """
#     model.train()
#     optimizer.zero_grad()

#     # === 1. 샘플링 ===
#     x0 = sampler.sample(batch_size, num_boxes)      # Gaussian 초기 박스
#     t = sampler.sample_t(batch_size)                # 시간 스칼라 [B]
#     t_broadcast = t.view(-1, 1, 1)                  # [B,1,1] for interpolation

#     # === 2. x1 (Target) = GT Box ===
#     # GT 수가 num_boxes보다 적을 수 있으므로 pad 필요
#     N_gt = gt_boxes.shape[1]
#     if N_gt < num_boxes:
#         pad = torch.zeros(batch_size, num_boxes - N_gt, 4, device=device)
#         x1 = torch.cat([gt_boxes, pad], dim=1)
#     else:
#         x1 = gt_boxes[:, :num_boxes]

#     # === 3. Interpolation ===
#     xt = sampler.sample_xt(x0, x1, t_broadcast)     # 중간 상태
#     ut = sampler.compute_target_flow(x0, x1)        # Target flow = x1 - x0

#     # === 4. 모델 예측 ===
#     pred_box, pred_obj = model(images, xt, t)

#     # === 5. Rectified Flow loss ===
#     flow_loss = F.mse_loss(pred_box, ut)

#     # === 6. Objectness loss (간단 버전) ===
#     target_obj = torch.ones_like(pred_obj)
#     obj_loss = F.mse_loss(pred_obj, target_obj)

#     # === 7. 총 손실 ===
#     total_loss = flow_loss + 0.1 * obj_loss
#     total_loss.backward()
#     optimizer.step()

#     return {
#         "total_loss": total_loss.item(),
#         "flow_loss": flow_loss.item(),
#         "obj_loss": obj_loss.item()
#     }



# =========================================================
#  Euler Integration 기반 추론 (inference_euler)
# =========================================================
@torch.no_grad()
def inference_euler(model, image, x0, steps=10, device="cuda"):
    """
    Rectified Flow 기반 Euler 적분 추론

    Args:
        model: FlowDet
        image: Tensor [B, 3, H, W]
        x0: Tensor [B, N, 4] - 초기 박스
        steps: int - Euler 적분 단계 수
    """
    model.eval()
    dt = 1.0 / steps
    xt = x0.clone().to(device)
    t = torch.zeros(x0.shape[0], device=device)

    for step in range(steps):
        # 예측된 flow (velocity)
        pred_box, _ = model(image, xt, t)

        # Euler update
        xt = xt + pred_box * dt
        t = t + dt

    return xt  # 최종 예측 (approx x1)
