# Setup stubs for optional heavy dependencies so the app can start
from app.utils.dependency_stubs import setup_dependency_stubs
setup_dependency_stubs()

# Import custom pandas initialization first to fix deprecation warnings
from app.utils.pandas_init import *

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json

from app.api.endpoints import predictions, models, data, backtesting, sentiment, auto_training, feedback
from app.utils.json_utils import CustomJSONEncoder

# Create a custom JSONResponse class that uses our CustomJSONEncoder
class CustomJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(
            content,
            cls=CustomJSONEncoder,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

app = FastAPI(
    title="CryptoPricer API",
    description="API for cryptocurrency price prediction using ML/DL models",
    version="0.1.0",
    default_response_class=CustomJSONResponse,  # Use our custom response class
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(backtesting.router, prefix="/api/backtesting", tags=["backtesting"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["sentiment"])
app.include_router(auto_training.router, prefix="/api/auto-training", tags=["auto-training"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])

# Start auto-training service on application startup
from app.services.auto_training_service import auto_training_service
import threading
import logging

logger = logging.getLogger(__name__)

@app.on_event("startup")
def startup_event():
    try:
        # Start auto-training in a background thread
        threading.Thread(target=auto_training_service.start_auto_training, daemon=True).start()
        logger.info("Auto-training service started successfully")

        # Start a background thread to process prediction feedback regularly
        from app.utils.prediction_feedback import prediction_feedback_system

        def process_feedback_regularly():
            import time
            while True:
                try:
                    logger.info("Processing prediction feedback")
                    prediction_feedback_system.process_feedback()
                    logger.info("Prediction feedback processed successfully")
                except Exception as e:
                    logger.error(f"Error processing prediction feedback: {str(e)}")

                # Sleep for 6 hours before processing again
                time.sleep(6 * 60 * 60)

        threading.Thread(target=process_feedback_regularly, daemon=True).start()
        logger.info("Prediction feedback processing service started successfully")
    except Exception as e:
        logger.error(f"Error starting services: {str(e)}")
        logger.error("Some services may not be available")

@app.on_event("shutdown")
def shutdown_event():
    # Stop auto-training when the application shuts down
    auto_training_service.stop_auto_training()

@app.get("/")
async def root():
    return {"message": "Welcome to CryptoPricer API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
