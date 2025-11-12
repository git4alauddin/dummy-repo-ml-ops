from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
import json
import time
from sklearn.datasets import load_iris

# OpenTelemetry for Tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --- 1. SETUP ---

# Setup Tracing to send data to Google Cloud Trace
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup Structured Logging to send machine-readable JSON to Google Cloud Logging
logger = logging.getLogger("iris-classifier-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s", "message": "%(message)s", "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI App & State
app = FastAPI()
app_state = {"is_ready": False, "model": None}
iris_target_names = load_iris().target_names

# --- 2. MODEL LOADING & HEALTH CHECKS ---

@app.on_event("startup")
async def startup_event():
    """On startup, load the model. The service is not "ready" until this is done."""
    logger.info(json.dumps({"event": "loading_model", "status": "started"}))
    app_state["model"] = joblib.load('iris_model.pkl')
    app_state["is_ready"] = True
    logger.info(json.dumps({"event": "loading_model", "status": "complete"}))

@app.get("/live_check", tags=["Health"])
async def liveness_probe():
    """Liveness probe: Is the app running?"""
    return {"status": "alive"}

@app.get("/ready_check", tags=["Health"])
async def readiness_probe():
    """Readiness probe: Is the app ready to take traffic? (i.e., is the model loaded?)"""
    if not app_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

# --- 3. PREDICTION ENDPOINT ---

class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

@app.post("/predict")
async def predict(input: IrisInput):
    """Main prediction endpoint."""
    with tracer.start_as_current_span("model_prediction") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = np.array([[
                input.sepal_length, input.sepal_width,
                input.petal_length, input.petal_width
            ]])
            
            prediction_idx = app_state["model"].predict(input_data)[0]
            prediction_class = iris_target_names[prediction_idx]

            latency = round((time.time() - start_time) * 1000, 2)
            result = {"prediction": prediction_class, "prediction_index": int(prediction_idx)}

            logger.info(json.dumps({
                "event": "prediction", "trace_id": trace_id,
                "input": input.dict(), "result": result, "latency_ms": latency,
            }))
            return result

        except Exception as e:
            logger.exception(json.dumps({"event": "prediction_error", "trace_id": trace_id, "error": str(e)}))
            raise HTTPException(status_code=500, detail="Prediction failed")
