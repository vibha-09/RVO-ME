from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routes import analysis
from .services.segmentation import load_segmentation_model
from .services.detection import load_detection_model

app = FastAPI(
    title="OCT Image Analysis API",
    description="API for analyzing OCT scan images for RVO-ME",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local testing, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Load models at startup
    load_segmentation_model()
    load_detection_model()

# Include routes
app.include_router(analysis.router, prefix="/api")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "OCT Analysis API is running."}
