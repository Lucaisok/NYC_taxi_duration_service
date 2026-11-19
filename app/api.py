from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.predict import load_model, predict
from src.schemas import DurationPrediction, TripFeatures


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: fetch the champion model from the MLflow registry
    app.state.model, app.state.model_version = load_model()
    yield
    app.state.model = None
    app.state.model_version = ""


app = FastAPI(lifespan=lifespan)


@app.get("/")
def get_root():
    return {"status": "OK"}


@app.post("/predict")
def get_trip_duration(trip: TripFeatures, req: Request) -> DurationPrediction:
    model = req.app.state.model
    duration = predict(model, trip.model_dump())
    return DurationPrediction(duration=f"{round(duration)} minutes")
