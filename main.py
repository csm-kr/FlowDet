import rootutils
rootutils.setup_root(__file__, indicator='.vscode', pythonpath=True)

import configargparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.flowdet import FlowDet
from loss.total_loss import HungarianMatcherDynamicK, FlowDetCriterion
from model.sampler import DistributionSampler
from dataset.voc_dataset import VOC_Dataset
from dataset import detection_transforms as det_transforms
from torchvision.ops import box_iou


# =========================================================
#  Euler Integration 기반 추론 (Rectified Flow)
# =========================================================
@torch.no_grad()
def inference_euler(model, image, x0, steps=10, device="cuda"):
    """
    Rectified Flow 기반 Euler 적분 추론
    """
    model.eval()
    dt = 1.0 / steps
    xt = x0.clone().to(device)
    t = torch.zeros(x0.shape[0], device=device)

    for step in range(steps):
        pred_box, _ = model(image, xt, t)
        xt = xt + pred_box * dt
        t = t + dt

    return xt  # [B, N, 4] 최종 예측 박스


# =========================================================
#  Argument Parser
# =========================================================
def get_args_parser():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        add_help=False
    )
    # parser.add_argument('--config', is_config_file=True, default='./configs/netmarble.yaml', help='config file path')
    # parser.add_argument('--dataset', type=str, default='netmarble', help='dataset name')
    parser.add_argument('--exp_name', type=str, default='flowdet_voc', help='experiment name')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--log_dir', type=str, default='./runs', help='tensorboard log dir')
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    return parser


# =========================================================
#  Validation Function
# =========================================================
@torch.no_grad()
def validate(model, criterion, sampler, val_loader, device="cuda"):
    model.eval()
    total_val_loss, total_iou = 0.0, 0.0
    count = 0

    for batch_idx, (images, boxes, labels, info) in enumerate(tqdm(val_loader, desc="Validation")):
        images = images.to(device)
        B = images.size(0)
        N_pred = 50

        # 샘플링
        x0 = sampler.sample(B, N_pred)
        pred_boxes = inference_euler(model, images, x0, steps=10, device=device)

        # GT 구성
        max_gt = max(len(b) for b in boxes)
        gt_boxes = torch.zeros((B, max_gt, 4), device=device)
        gt_obj = torch.zeros((B, max_gt), device=device)
        for i, b in enumerate(boxes):
            L = len(b)
            gt_boxes[i, :L] = b.to(device)
            gt_obj[i, :L] = 1.0

        # 로스 계산
        outputs = {"pred_boxes": pred_boxes, "pred_objectness": torch.ones(B, N_pred, device=device)}
        targets = {"boxes": gt_boxes, "objectness": gt_obj}
        losses = criterion(outputs, targets)
        total_val_loss += losses["loss_total"].item()

        # IoU 계산 (batch 평균)
        for i in range(B):
            ious = box_iou(pred_boxes[i], gt_boxes[i][:len(boxes[i])])
            total_iou += ious.max(dim=0).values.mean().item()
            count += 1

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_iou = total_iou / max(1, count)
    return avg_val_loss, avg_val_iou


# =========================================================
#  Training Main
# =========================================================
def train_main(args):
    device = args.device

    writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.exp_name}")
    print(f"📊 TensorBoard logging to: {args.log_dir}/{args.exp_name}")

    # -------------------------------
    # Dataset & DataLoader
    # -------------------------------
    transform_train = det_transforms.DetCompose([
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=(224, 224), box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    transform_val = det_transforms.DetCompose([
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=(224, 224), box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VOC_Dataset(root=r"D:\data\voc", split="train", download=False, transform=transform_train, visualization=False)
    val_dataset = VOC_Dataset(root=r"D:\data\voc", split="test", download=False, transform=transform_val, visualization=False)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=val_dataset.collate_fn, shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------------
    # Model, Loss, Optimizer
    # -------------------------------
    model = FlowDet().to(device)
    matcher = HungarianMatcherDynamicK(cost_class=1, cost_bbox=2, cost_giou=1, topk=5)
    criterion = FlowDetCriterion(matcher, weight_cls=1.0, weight_box=2.0)
    sampler = DistributionSampler(out_dim=4, device=device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # -------------------------------
    # Training Loop
    # -------------------------------
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, boxes, labels) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}/{args.epochs}]")):
        # for batch_idx, (images, boxes, labels) in enumerate(tqdm(val_loader, desc=f"[Epoch {epoch+1}/{args.epochs}]")):
            images = images.to(device)
            B = images.size(0)
            N_pred = 50

            # 샘플링
            x0 = sampler.sample(B, N_pred)
            t = sampler.sample_t(B)

            # Forward
            pred_box, pred_objectness = model(images, x0, t)

            # GT 구성
            max_gt = max(len(b) for b in boxes)
            gt_boxes = torch.zeros((B, max_gt, 4), device=device)
            gt_obj = torch.zeros((B, max_gt), device=device)
            for i, b in enumerate(boxes):
                L = len(b)
                gt_boxes[i, :L] = b.to(device)
                gt_obj[i, :L] = 1.0

            outputs = {"pred_boxes": pred_box, "pred_objectness": pred_objectness.squeeze(-1)}
            targets = {"boxes": gt_boxes, "objectness": gt_obj}

            # Loss
            losses = criterion(outputs, targets)
            total_loss = losses["loss_total"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # TensorBoard logging
            writer.add_scalar("train/loss_total", losses["loss_total"], global_step)
            writer.add_scalar("train/loss_cls", losses["loss_cls"], global_step)
            writer.add_scalar("train/loss_box", losses["loss_box"], global_step)
            epoch_loss += total_loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # -------------------------------
        # Validation (inference)
        # -------------------------------
        avg_val_loss, avg_val_iou = validate(model, criterion, sampler, val_loader, device)
        print(f"Validation Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}")

        writer.add_scalar("val/loss_total", avg_val_loss, epoch)
        writer.add_scalar("val/mean_iou", avg_val_iou, epoch)

    writer.close()
    print("✅ Training + Validation finished!")


# =========================================================
#  Entry Point
# =========================================================
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train_main(args)
