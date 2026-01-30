import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditScoringRequest, CreditScoringResponse
from src.predict import CreditScoringModel
import mlflow.sklearn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreditRiskAPI")

# Global Variable
model_engine = None


# --- Lifespan Logic (Replaces @app.on_event) ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Everything before 'yield' runs on startup.
    Everything after 'yield' runs on shutdown.
    """
    global model_engine
    try:
        model_uri = "models:/CreditRisk_LogisticRegression/Latest"
        model_engine = CreditScoringModel(model_uri)
        logger.info(f"Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        model_engine = None

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="Bati Bank Credit Scoring API",
    description="Microservice for predicting credit risk.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    status = "healthy" if model_engine else "degraded"
    return {"status": status}

@app.post("/predict", response_model=CreditScoringResponse)
def predict_risk(customer_id: str, request: CreditScoringRequest, is_raw: bool = True):
    """
    Credit scoring endpoint.
    
    Args:
        customer_id: Unique customer identifier
        request: CreditScoringRequest with customer features
        is_raw: Whether input is raw customer data (True) or already scaled (False)
    
    Returns:
        CreditScoringResponse with credit score, probability of default, and approval status
    """
    if not model_engine:
        raise HTTPException(status_code=503, detail="Scoring model is not initialized.")
    
    try:
        # Pydantic v2 uses model_dump() instead of dict()
        input_data = request.model_dump(exclude_none=True)
        
        logger.info(f"Processing request for customer {customer_id}. Is_raw={is_raw}")
        logger.info(f"Input features: {list(input_data.keys())}")
        
        # Call predict with the is_raw flag
        result = model_engine.predict(input_data, is_raw=is_raw)
        is_approved = result['credit_score'] >= 650

        return CreditScoringResponse(
            customer_id=customer_id,
            probability_of_default=result['probability_of_default'],
            credit_score=result['credit_score'],
            risk_tier=result['risk_tier'],
            approved=is_approved
        )
    except Exception as e:
        logger.error(f"Prediction error for customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))