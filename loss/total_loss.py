import rootutils
rootutils.setup_root(__file__, indicator='.vscode', pythonpath=True)
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou


class HungarianMatcherDynamicK(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1, topk=5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.topk = topk

    @torch.no_grad()
    def forward(self, pred_box, pred_obj, gt_box, gt_obj):
        """
        pred_box: [B, N_pred, 4]
        pred_obj: [B, N_pred]
        gt_box: [B, N_gt, 4]
        gt_obj: [B, N_gt]
        """
        bs, num_pred = pred_box.shape[:2]
        indices = []

        for b in range(bs):
            pb, gb = pred_box[b], gt_box[b]
            # po, go = pred_obj[b].sigmoid(), gt_obj[b]

            # --- cost 계산 ---
            # cost_class = -po[:, None] * go[None, :]         # 낮을수록 좋음
            cost_bbox = torch.cdist(pb, gb, p=1)
            cost_giou = 1 - generalized_box_iou(pb, gb)

            # C = self.cost_class * cost_class + \
            C = self.cost_bbox * cost_bbox + \
                self.cost_giou * cost_giou

            # --- GT마다 top-K 예측 선택 ---
            C = C.cpu()
            num_gt = gb.shape[0]
            matched_pred, matched_gt = [], []
            for j in range(num_gt):
                topk_idx = torch.topk(C[:, j], self.topk, largest=False).indices
                matched_pred.extend(topk_idx.tolist())
                matched_gt.extend([j] * self.topk)

            indices.append((torch.as_tensor(matched_pred), torch.as_tensor(matched_gt)))

        return indices



def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """
    inputs: [N,] raw logits
    targets: [N,] binary labels
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def flow_loss(pred_box, gt_box):
    """
    Rectified flow matching (L2)
    """
    return F.mse_loss(pred_box, gt_box)

class FlowDetCriterion(nn.Module):
    def __init__(self, matcher, weight_cls=1.0, weight_box=2.0):
        super().__init__()
        self.matcher = matcher
        self.weight_cls = weight_cls
        self.weight_box = weight_box

    def forward(self, outputs, targets):
        """
        outputs: dict {
            'pred_boxes': [B, N_pred, 4],
            'pred_objectness': [B, N_pred],
        }
        targets: dict {
            'boxes': [B, N_gt, 4],
            'objectness': [B, N_gt],
        }
        """
        pred_box = outputs["pred_boxes"]
        pred_obj = outputs["pred_objectness"]
        gt_box = targets["boxes"]
        gt_obj = targets["objectness"]

        # 1. Matching
        indices = self.matcher(pred_box, pred_obj, gt_box, gt_obj)

        total_cls_loss, total_box_loss = 0, 0
        for b, (pred_idx, gt_idx) in enumerate(indices):
            pb, gb = pred_box[b][pred_idx], gt_box[b][gt_idx]
            # po, go = pred_obj[b][pred_idx], gt_obj[b][gt_idx]

            # Focal Loss (objectness)
            # cls_loss = focal_loss(po, go)

            # Flow-matching bbox regression
            box_loss = flow_loss(pb, gb)

            # total_cls_loss += cls_loss
            total_box_loss += box_loss

        total_cls_loss /= len(indices)
        total_box_loss /= len(indices)

        total_loss = self.weight_cls * total_cls_loss + self.weight_box * total_box_loss

        return {
            "loss_total": total_loss,
            "loss_cls": total_box_loss,
            "loss_box": total_box_loss,
        }
    
if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== 1. 더미 데이터 생성 ====
    B = 2          # batch size
    N_pred = 50    # predicted boxes
    N_gt = 7       # ground-truth boxes per image

    # 모델 출력 (FlowDet)
    outputs = {
        "pred_boxes": torch.rand(B, N_pred, 4, device=device),         # [0,1] normalized
        "pred_objectness": torch.randn(B, N_pred, device=device)       # raw logits
    }

    # GT (VOC 등에서 온 박스)
    targets = {
        "boxes": torch.rand(B, N_gt, 4, device=device),
        "objectness": torch.ones(B, N_gt, device=device)
    }

    # ==== 2. Criterion & Matcher 초기화 ====
    matcher = HungarianMatcherDynamicK(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0, topk=5)
    criterion = FlowDetCriterion(matcher, weight_cls=1.0, weight_box=2.0).to(device)

    # ==== 3. Forward ====
    losses = criterion(outputs, targets)

    # ==== 4. 출력 ====
    print("=== Criterion Output ===")
    for k, v in losses.items():
        print(f"{k}: {v:.6f}")