import torch
import torch.nn as nn

class DistributionSampler(nn.Module):
    """
    Gaussian 전용 Distribution Sampler (FlowDet용)
    - Rectified Flow 학습용
    - x0만 샘플링, t는 외부 또는 sample_t()로 별도 생성
    """

    def __init__(self, out_dim=4, device='cuda'):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

    # =========================================================
    # x0 샘플링
    # =========================================================
    def sample(self, batch_size, num_boxes):
        """
        Gaussian 분포에서 x0 샘플링
        Args:
            batch_size: B
            num_boxes: N (object 수)
        Returns:
            x0: [B, N, D]
        """
        x0 = torch.randn(batch_size, num_boxes, self.out_dim, device=self.device)
        return x0

    # =========================================================
    # t만 별도 샘플링
    # =========================================================
    def sample_t(self, batch_size):
        """
        interpolation 비율 t만 별도로 샘플링
        Returns:
            t: [B]
        """
        return torch.rand(batch_size, device=self.device)

    # =========================================================
    # Interpolation (x_t)
    # =========================================================
    def sample_xt(self, x0, x1, t):
        """
        Linear interpolation:
        x_t = (1 - t) * x0 + t * x1
        t: [B] → 자동 broadcast
        """
        t = t.view(-1, 1, 1)  # [B,1,1]로 reshape하여 broadcast
        return (1 - t) * x0 + t * x1

    # =========================================================
    # Flow target (u_t)
    # =========================================================
    def compute_target_flow(self, x0, x1):
        """
        Rectified flow target: u_t = x1 - x0
        """
        return x1 - x0

    # =========================================================
    # 데이터 범위 전처리 (선택적)
    # =========================================================
    def preprocess(self, data, reverse=False):
        """
        [-1,1] <-> [0,1] 변환
        """
        if not reverse:
            return 2 * data - 1
        else:
            return (data + 1) / 2


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = DistributionSampler(out_dim=4, device=device)

    batch_size = 2
    num_boxes = 50

    # 샘플링
    x0 = sampler.sample(batch_size, num_boxes)
    t = sampler.sample_t(batch_size)
    x1 = torch.randn(batch_size, num_boxes, 4, device=device)  # target
    xt = sampler.sample_xt(x0, x1, t)
    ut = sampler.compute_target_flow(x0, x1)

    print("x0:", x0.shape)  # [2, 50, 4]
    print("x1:", x1.shape)  # [2, 50, 4]
    print("t:", t.shape)    # [2]
    print("xt:", xt.shape)  # [2, 50, 4]
    print("ut:", ut.shape)  # [2, 50, 4]
