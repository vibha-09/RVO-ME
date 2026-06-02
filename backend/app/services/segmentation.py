import torch
import numpy as np
import cv2
from ..models.attention_unet import AttentionUNet
import os

model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_segmentation_model(weights_path="weights/attention_unet.pth"):
    global model
    model = AttentionUNet(img_ch=1, output_ch=5).to(device)
    
    # Try to load weights if they exist, else we use randomly initialized model for stubbing
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded Attention U-Net weights.")
    else:
        print(f"Warning: {weights_path} not found. Running with randomly initialized weights (Stub Mode).")
    
    model.eval()

def predict_segmentation(enhanced_image: np.ndarray) -> np.ndarray:
    """
    Runs the segmentation model on the preprocessed grayscale image.
    Returns a class mask (H, W) where each pixel is an integer class label (0-4).
    """
    global model
    if model is None:
        load_segmentation_model()
        
    # Resize to standard size (e.g., 256x256)
    target_size = (256, 256)
    orig_h, orig_w = enhanced_image.shape[:2]
    
    img_resized = cv2.resize(enhanced_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor, add batch and channel dims -> (1, 1, H, W)
    input_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Apply argmax to get the class with highest probability
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize back to original dimensions using Nearest Neighbor
    predicted_mask_orig_size = cv2.resize(predicted_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return predicted_mask_orig_size


def predict_segmentation_proba(enhanced_image: np.ndarray) -> np.ndarray:
    """
    Returns per-pixel class probabilities (H, W, N_CLASSES) float32.
    Used for ROC/AUC computation in evaluation.
    """
    global model
    if model is None:
        load_segmentation_model()

    target_size = (256, 256)
    orig_h, orig_w = enhanced_image.shape[:2]

    img_resized = cv2.resize(enhanced_image, target_size, interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

    import torch.nn.functional as F
    with torch.no_grad():
        logits = model(input_tensor)                         # (1, 5, 256, 256)
        probs  = F.softmax(logits, dim=1).squeeze(0)        # (5, 256, 256)
        probs_np = probs.cpu().numpy().astype(np.float32)   # (5, H, W)

    # Resize each class channel back to original size
    out = np.zeros((5, orig_h, orig_w), dtype=np.float32)
    for c in range(5):
        out[c] = cv2.resize(probs_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return out.transpose(1, 2, 0)   # (H, W, 5)
