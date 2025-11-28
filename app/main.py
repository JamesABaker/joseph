"""
FastAPI application for AI text detection.
Serves both REST API and web UI.
"""

import logging
import os
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.dependencies import get_current_user, get_current_user_optional
from app.ml_model import AIDetector
from app.models import Result, User
from app.routes import auth as auth_routes

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

# Include auth routes
app.include_router(auth_routes.router)

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
    """Load model and create database tables on startup."""
    global detector
    logger.info("Starting application...")
    try:
        # Create database tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created")

        # Load ML model
        detector = AIDetector()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
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


@app.get("/tos", response_class=HTMLResponse)
def tos():
    """Serve the terms of service page."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "tos.html")
    try:
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>TOS not found</h1><p>Please ensure static/tos.html exists.</p>",
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
def detect_text(
    request: DetectionRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Detect if text is AI-generated or human-written.
    Authentication is optional. If authenticated, saves result to user's history.

    Args:
        request: DetectionRequest containing the text to analyze
        current_user: Authenticated user from JWT token (optional)
        db: Database session

    Returns:
        DetectionResponse with probabilities and prediction
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = detector.detect(request.text)
        result["text_length"] = len(request.text)

        # Convert numpy types to Python native types for database compatibility
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            import numpy as np

            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        # Save result to database only if user is authenticated
        if current_user:
            db_result = Result(
                user_id=current_user.id,
                text_analyzed=request.text,
                human_probability=convert_numpy(result["human_probability"]),
                ai_probability=convert_numpy(result["ai_probability"]),
                prediction=result["prediction"],
                ml_human_probability=convert_numpy(result["ml_human_probability"]),
                ml_ai_probability=convert_numpy(result["ml_ai_probability"]),
                perplexity=convert_numpy(result["perplexity"]),
                shannon_entropy=convert_numpy(result["shannon_entropy"]),
                burstiness=convert_numpy(result["burstiness"]),
                lexical_diversity=convert_numpy(result["lexical_diversity"]),
                word_length_variance=convert_numpy(result["word_length_variance"]),
                punctuation_diversity=convert_numpy(result["punctuation_diversity"]),
                vocabulary_richness=convert_numpy(result["vocabulary_richness"]),
                entropy_ai_probability=convert_numpy(result["entropy_ai_probability"]),
                entropy_human_probability=convert_numpy(result["entropy_human_probability"]),
            )
            db.add(db_result)
            db.commit()

        return DetectionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during detection")


@app.get("/api/results")
async def get_user_results(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """
    Get detection results history for the current user.

    Args:
        current_user: Authenticated user from JWT token
        db: Database session
        limit: Maximum number of results to return (default 50)
        offset: Number of results to skip (default 0)

    Returns:
        List of user's detection results
    """
    results = (
        db.query(Result)
        .filter(Result.user_id == current_user.id)
        .order_by(Result.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )

    return {
        "results": [
            {
                "id": r.id,
                "text_analyzed": (
                    r.text_analyzed[:100] + "..." if len(r.text_analyzed) > 100 else r.text_analyzed
                ),
                "human_probability": r.human_probability,
                "ai_probability": r.ai_probability,
                "prediction": r.prediction,
                "created_at": r.created_at.isoformat(),
            }
            for r in results
        ],
        "total": db.query(Result).filter(Result.user_id == current_user.id).count(),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/results/{result_id}")
async def get_result_detail(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get detailed information for a specific result.

    Args:
        result_id: ID of the result to retrieve
        current_user: Authenticated user from JWT token
        db: Database session

    Returns:
        Detailed result information
    """
    result = (
        db.query(Result).filter(Result.id == result_id, Result.user_id == current_user.id).first()
    )

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    return {
        "id": result.id,
        "text_analyzed": result.text_analyzed,
        "human_probability": result.human_probability,
        "ai_probability": result.ai_probability,
        "prediction": result.prediction,
        "ml_human_probability": result.ml_human_probability,
        "ml_ai_probability": result.ml_ai_probability,
        "perplexity": result.perplexity,
        "shannon_entropy": result.shannon_entropy,
        "burstiness": result.burstiness,
        "lexical_diversity": result.lexical_diversity,
        "word_length_variance": result.word_length_variance,
        "punctuation_diversity": result.punctuation_diversity,
        "vocabulary_richness": result.vocabulary_richness,
        "entropy_ai_probability": result.entropy_ai_probability,
        "entropy_human_probability": result.entropy_human_probability,
        "created_at": result.created_at.isoformat(),
    }


@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests."""
    return HTMLResponse(content="", status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
