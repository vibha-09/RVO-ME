import cv2
import numpy as np

def create_overlay(image: np.ndarray, mask: np.ndarray, detections: dict) -> np.ndarray:
    """
    Creates an overlay visualization of the segmentation mask and bounding boxes.
    """
    # Ensure image is RGB for drawing colored overlays
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        overlay = image.copy()
        
    h, w = mask.shape
    
    # Define colors for classes (BGR if using cv2 directly, but let's use RGB for PIL/matplotlib compatibility later, sticking to RGB)
    # Using RGB: 1: SRF (Blue), 2: IRF (Red), 3: ELM (Yellow), 4: EZ (Green)
    colors = {
        1: [0, 0, 255],    # Blue
        2: [255, 0, 0],    # Red
        3: [255, 255, 0],  # Yellow
        4: [0, 255, 0]     # Green
    }
    
    mask_colored = np.zeros_like(overlay)
    
    for cls, color in colors.items():
        mask_colored[mask == cls] = color
        
    # Alpha blend the mask onto the image
    alpha = 0.4
    overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)
    
    # Draw bounding boxes (Cyan for HF)
    for box in detections.get('boxes', []):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
    return overlay
