from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

from ..services.preprocessor import apply_clahe
from ..services.segmentation import predict_segmentation
from ..services.detection import predict_detections
from ..services.quantification import quantify_biomarkers
from ..services.pipeline import extract_features, predict_pipeline
from ..utils.visualization import create_overlay

router = APIRouter()

def encode_image_base64(img_array: np.ndarray) -> str:
    """Helper to encode a numpy array image to base64 string."""
    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array)
    else:
        # Convert BGR to RGB for PIL if it's 3 channel from openCV (though we tried keeping RGB)
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@router.post("/analyze")
async def analyze_oct(file: UploadFile = File(...)):
    """
    Main endpoint for analyzing an OCT image.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read the file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Load as RGB conceptually
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure RGB
        
        # 1. Preprocess
        enhanced_image = apply_clahe(image)
        
        # 2. Segment
        segmentation_mask = predict_segmentation(enhanced_image)
        
        # 3. Detect
        detections = predict_detections(enhanced_image)
        
        # 4. Biomarker Quantification
        metrics = quantify_biomarkers(segmentation_mask, detections)

        # 5. GMM phenotyping — cluster assignment driven by the actual data
        #    distribution of your patient population (train_phenotyping.py).
        features    = extract_features(segmentation_mask, detections)
        phenotyping = predict_pipeline(features)

        # 6. Overlay Visualization
        overlay = create_overlay(image, segmentation_mask, detections)

        # Encode images for JSON response
        results = {
            "original_image":    f"data:image/png;base64,{encode_image_base64(image)}",
            "enhanced_image":    f"data:image/png;base64,{encode_image_base64(enhanced_image)}",
            "segmentation_mask": f"data:image/png;base64,{encode_image_base64(create_overlay(np.zeros_like(image), segmentation_mask, {}))}",
            "overlay_image":     f"data:image/png;base64,{encode_image_base64(overlay)}",
            "metrics":           metrics,
            "phenotyping":       phenotyping,
        }
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
