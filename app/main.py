"""
FastAPI application for AI text detection.
Serves both REST API and web UI.
"""
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.model import AIDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Joseph - AI Text Detector API",
    description="Ensemble analysis of ML and entropy-based methods to detect AI-generated text.",
    version="2.0.0",
)

# Initialize model (singleton - loaded once at startup)
detector: Optional[AIDetector] = None


class DetectionRequest(BaseModel):
    """Request model for text detection."""

    text: str = Field(..., min_length=1, description="Text to analyze")

    class Config:
        json_schema_extra = {
            "example": {"text": "The rapid advancement of AI has transformed industries."}
        }


class DetectionResponse(BaseModel):
    """Response model for detection results."""

    # Final hybrid scores
    human_probability: float = Field(..., description="Probability text is human-written (0-100)")
    ai_probability: float = Field(..., description="Probability the text is AI-generated (0-100)")
    prediction: str = Field(..., description="Final prediction: 'human' or 'ai'")
    text_length: int = Field(..., description="Length of analyzed text in characters")

    # ML model scores
    ml_human_probability: float = Field(..., description="ML model human probability")
    ml_ai_probability: float = Field(..., description="ML model AI probability")

    # Entropy metrics
    perplexity: float = Field(..., description="Text perplexity score")
    shannon_entropy: float = Field(..., description="Shannon entropy of text")
    burstiness: float = Field(..., description="Sentence complexity variation (0-1)")
    lexical_diversity: float = Field(..., description="Unique word ratio (0-1)")
    word_length_variance: float = Field(..., description="Word length variance (0-1)")
    punctuation_diversity: float = Field(..., description="Punctuation diversity (0-1)")
    vocabulary_richness: float = Field(..., description="Vocabulary richness (0-1)")
    entropy_ai_probability: float = Field(..., description="Entropy-based AI probability")
    entropy_human_probability: float = Field(..., description="Entropy-based human probability")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global detector
    logger.info("Starting application...")
    try:
        detector = AIDetector()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the web UI."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>UI not found</h1><p>Please ensure static/index.html exists.</p>",
            status_code=404,
        )


@app.get("/about", response_class=HTMLResponse)
def about():
    """Serve the about page."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "about.html")
    try:
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>About page not found</h1><p>Please ensure static/about.html exists.</p>",
            status_code=404,
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": detector is not None}


@app.get("/api/model-info")
async def model_info():
    """Get information about the detection model."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return detector.get_model_info()


@app.post("/api/detect", response_model=DetectionResponse)
def detect_text(request: DetectionRequest):
    """
    Detect if text is AI-generated or human-written.

    Args:
        request: DetectionRequest containing the text to analyze

    Returns:
        DetectionResponse with probabilities and prediction
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = detector.detect(request.text)
        result["text_length"] = len(request.text)
        return DetectionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during detection")


@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests."""
    return HTMLResponse(content="", status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
