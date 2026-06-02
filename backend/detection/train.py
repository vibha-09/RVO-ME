"""
Faster R-CNN training script for Hyperreflective Foci (HF) detection.

Designed to run on Kaggle (GPU) or any machine with CUDA/CPU.
After training the best model weights (hf_frcnn.pt) and a companion config
(hf_frcnn_config.json) are saved to --output_dir.  Both files are required by
inference.py for identical architecture reconstruction.

Usage (Kaggle / CLI):
    python train.py \
        --coco_json  /kaggle/working/coco_output/annotations.json \
        --images_dir /kaggle/working/coco_output/images \
        --output_dir /kaggle/working
"""

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou

from dataset import HFDetectionDataset, collate_fn, get_transform


# ─── Default training config ──────────────────────────────────────────────────
# All values are serialised to hf_frcnn_config.json and loaded by inference.py.
# Do NOT change these after training unless you retrain from scratch.

DEFAULT_CONFIG: dict = {
    # Model / architecture
    "num_classes":          2,          # background + HF
    "image_min_size":       800,
    "image_max_size":       1333,
    # Small-object anchors: same anchor count per level (3) as COCO defaults
    # so the RPN head weight shapes are state_dict-compatible with pretrained weights.
    "anchor_sizes":         [[8], [16], [32], [64], [128]],
    "anchor_aspect_ratios": [[0.5, 1.0, 2.0]] * 5,
    # Preprocessing (informational – actual logic lives in dataset.preprocess_image)
    "clahe_clip_limit":     2.0,
    "clahe_tile_grid":      [8, 8],
    # Inference
    "score_threshold":      0.5,
    # Training
    "seed":                 42,
    "lr":                   1e-4,
    "weight_decay":         1e-4,
    "lr_step_size":         10,
    "lr_gamma":             0.5,
    "grad_clip_norm":       5.0,
    "epochs":               30,
    "batch_size":           2,
    "val_split":            0.2,
    "num_workers":          2,
}


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Model builder ───────────────────────────────────────────────────────────

def build_model(config: dict, pretrained: bool = True) -> torch.nn.Module:
    """
    Construct Faster R-CNN with small-object-optimised anchors.

    CRITICAL: This exact function (with the same config) must be called in
    inference.py to reconstruct the architecture before loading state_dict.
    Any change here requires retraining.
    """
    anchor_gen = AnchorGenerator(
        sizes=tuple(tuple(s) for s in config["anchor_sizes"]),
        aspect_ratios=tuple(tuple(r) for r in config["anchor_aspect_ratios"]),
    )

    # torchvision >= 0.13 uses weights=; older uses pretrained=
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=config["image_min_size"],
            max_size=config["image_max_size"],
            rpn_anchor_generator=anchor_gen,
        )
    except ImportError:
        model = fasterrcnn_resnet50_fpn(  # type: ignore[call-arg]
            pretrained=pretrained,
            min_size=config["image_min_size"],
            max_size=config["image_max_size"],
            rpn_anchor_generator=anchor_gen,
        )

    # Replace classification head for our 2-class task
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])
    return model


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:           torch.nn.Module,
    loader:          DataLoader,
    device:          torch.device,
    iou_threshold:   float = 0.5,
    score_threshold: float = 0.5,
) -> dict:
    """
    Compute Precision, Recall, and F1 at a fixed IoU threshold.

    Boxes are matched greedily from highest to lowest score (standard AP
    matching).  Returns a dict with keys precision / recall / f1.
    """
    model.eval()
    tp = fp = fn = 0

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_boxes  = tgt["boxes"].to(device)
            pred_mask = out["scores"] >= score_threshold
            pred_boxes   = out["boxes"][pred_mask]
            pred_scores  = out["scores"][pred_mask]

            n_gt   = gt_boxes.shape[0]
            n_pred = pred_boxes.shape[0]

            if n_gt == 0 and n_pred == 0:
                continue
            if n_gt == 0:
                fp += n_pred
                continue
            if n_pred == 0:
                fn += n_gt
                continue

            iou_mat    = box_iou(pred_boxes, gt_boxes)   # [n_pred, n_gt]
            matched_gt = set()
            order      = pred_scores.argsort(descending=True)

            for pi in order:
                best_iou, best_gi = iou_mat[pi].max(dim=0)
                if best_iou >= iou_threshold and best_gi.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gi.item())
                else:
                    fp += 1
            fn += n_gt - len(matched_gt)

    eps       = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(
    model:     torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loader:    DataLoader,
    device:    torch.device,
    epoch:     int,
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device)                           for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()

        running_loss += loss.item()

        if i % 10 == 0:
            parts = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  [E{epoch:03d} B{i:04d}] total={loss.item():.4f}  {parts}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def compute_val_loss(
    model:   torch.nn.Module,
    loader:  DataLoader,
    device:  torch.device,
) -> float:
    """Val loss requires train mode (Faster R-CNN only computes losses in train mode)."""
    model.train()
    total = 0.0
    for images, targets in loader:
        images  = [img.to(device)                           for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total  += sum(model(images, targets).values()).item()
    model.eval()
    return total / max(len(loader), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Merge args into config ────────────────────────────────────────────────
    config = deepcopy(DEFAULT_CONFIG)
    config.update({
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "val_split":      args.val_split,
        "seed":           args.seed,
        "image_min_size": args.min_size,
        "score_threshold":args.score_threshold,
        "num_workers":    args.num_workers,
    })

    set_seed(config["seed"])
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device     : {device}")
    print(f"Output dir : {output_dir}")

    # ── Build datasets ────────────────────────────────────────────────────────
    all_indices = list(range(
        len(HFDetectionDataset(args.coco_json, args.images_dir))
    ))
    random.shuffle(all_indices)
    split      = int(len(all_indices) * (1 - config["val_split"]))
    train_idx  = all_indices[:split]
    val_idx    = all_indices[split:]
    print(f"Train : {len(train_idx)}  |  Val : {len(val_idx)}")

    train_ds = HFDetectionDataset(args.coco_json, args.images_dir,
                                  transforms=get_transform(train=True))
    val_ds   = HFDetectionDataset(args.coco_json, args.images_dir,
                                  transforms=get_transform(train=False))

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # ── Build model + optimiser ───────────────────────────────────────────────
    model     = build_model(config, pretrained=True).to(device)
    params    = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(params, lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser,
        step_size=config["lr_step_size"],
        gamma=config["lr_gamma"],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    weights_path   = output_dir / "hf_frcnn.pt"
    config_path    = output_dir / "hf_frcnn_config.json"
    history: list[dict] = []

    for epoch in range(1, config["epochs"] + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, optimiser, train_loader, device,
                                     epoch, config["grad_clip_norm"])
        v_loss     = compute_val_loss(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{config['epochs']} | "
              f"train={train_loss:.4f}  val={v_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": v_loss})

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), weights_path)
            print(f"  ✓ New best – saved ({weights_path.name})")

    # ── Save config and history ───────────────────────────────────────────────
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    with open(output_dir / "train_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\nConfig  : {config_path}")
    print(f"Weights : {weights_path}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n─── Final evaluation on validation set ───")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    metrics = evaluate(model, val_loader, device,
                       score_threshold=config["score_threshold"])
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}" if isinstance(v, float) else f"  {k:<12}: {v}")

    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nAll artefacts saved to {output_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train Faster R-CNN for HF detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coco_json",      required=True)
    p.add_argument("--images_dir",     required=True)
    p.add_argument("--output_dir",     default="./output")
    p.add_argument("--lr",             type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--weight_decay",   type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--epochs",         type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size",     type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--val_split",      type=float, default=DEFAULT_CONFIG["val_split"])
    p.add_argument("--seed",           type=int,   default=DEFAULT_CONFIG["seed"])
    p.add_argument("--min_size",       type=int,   default=DEFAULT_CONFIG["image_min_size"])
    p.add_argument("--score_threshold",type=float, default=DEFAULT_CONFIG["score_threshold"])
    p.add_argument("--num_workers",    type=int,   default=DEFAULT_CONFIG["num_workers"])
    main(p.parse_args())
