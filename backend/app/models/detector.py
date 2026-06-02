"""
Faster R-CNN model factory for HF detection.

Anchor sizes are tuned for small objects (HF lesions).  Each FPN level still
uses 3 anchors per location (1 size × 3 aspect ratios), keeping the RPN head
weight shapes identical to the COCO-pretrained defaults so that pretrained
weights load without shape mismatches.

The same architecture is reproduced in backend/detection/train.py and
backend/detection/inference.py.  Any change here requires retraining.
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


# ─── Architecture constants (must match detection/train.py DEFAULT_CONFIG) ────

_ANCHOR_SIZES         = ((8,), (16,), (32,), (64,), (128,))
_ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5
_IMAGE_MIN_SIZE       = 800
_IMAGE_MAX_SIZE       = 1333


def get_detection_model(num_classes: int = 2) -> torchvision.models.detection.FasterRCNN:
    """
    Build Faster R-CNN with small-object anchors.

    Args:
        num_classes: Total classes including background (default 2: bg + HF).

    Returns:
        Model with pretrained backbone; classification head replaced for
        num_classes outputs.
    """
    anchor_gen = AnchorGenerator(
        sizes=_ANCHOR_SIZES,
        aspect_ratios=_ANCHOR_ASPECT_RATIOS,
    )

    # torchvision >= 0.13 uses weights=; older uses pretrained=
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            min_size=_IMAGE_MIN_SIZE,
            max_size=_IMAGE_MAX_SIZE,
            rpn_anchor_generator=anchor_gen,
        )
    except ImportError:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(  # type: ignore[call-arg]
            pretrained=True,
            min_size=_IMAGE_MIN_SIZE,
            max_size=_IMAGE_MAX_SIZE,
            rpn_anchor_generator=anchor_gen,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
