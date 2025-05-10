from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
import numpy as np
from pydantic import BaseModel, Field
import uvicorn
import pickle


# Load models
try:
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("models/svc_model.pkl", "rb") as f:
        svc = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")


app = FastAPI(
    title="Pistachio Classification API",
    description="Classifying pistachios using an ML model served with FastAPI",
)


class PistachioFeatures(BaseModel):
    Area: int = Field(ge=29808, le=124008, example=79950)
    Perimeter: float = Field(ge=858.363, le=2755.049, example=1425.97)
    Major_axis: float = Field(ge=320.3445, le=541.9661, example=446.25)
    Minor_axis: float = Field(ge=133.5096, le=383.0461, example=238.31)
    Eccentricity: float = Field(ge=0.5049, le=0.946, example=0.84)
    Eqdiasq: float = Field(ge=194.8146, le=397.3561, example=317.92)
    Solidity: float = Field(ge=0.588, le=0.9951, example=0.94)
    Convex_area: int = Field(ge=37935, le=132478, example=85015)
    Extent: float = Field(ge=0.4272, le=0.8204, example=0.716)
    Aspect_ratio: float = Field(ge=1.1585, le=3.0858, example=1.898)
    Roundness: float = Field(ge=0.0628, le=0.9336, example=0.569)
    Compactness: float = Field(ge=0.476, le=0.8779, example=0.713)
    Shapefactor_1: float = Field(ge=0.004, le=0.0131, example=0.0057)
    Shapefactor_2: float = Field(ge=0.0024, le=0.0053, example=0.0030)
    Shapefactor_3: float = Field(ge=0.2266, le=0.7706, example=0.510)
    Shapefactor_4: float = Field(ge=0.6204, le=0.999, example=0.955)


class ResponseModel(BaseModel):
    Predict: int
    Pistachio_type: str


@app.post("/predict", response_model=ResponseModel, status_code=status.HTTP_200_OK)
async def predict(input: PistachioFeatures):
    """Classifies pistachios and returns their type."""

    if scaler is None or pca is None or svc is None:
        raise HTTPException(status_code=500, detail="ML model error. Please try again later.")

    # Convert input to NumPy array
    try:
        input_data = np.array([[input.Area, input.Perimeter, input.Major_axis, input.Minor_axis,
                                input.Eccentricity, input.Eqdiasq, input.Solidity, input.Convex_area,
                                input.Extent, input.Aspect_ratio, input.Roundness, input.Compactness,
                                input.Shapefactor_1, input.Shapefactor_2, input.Shapefactor_3, input.Shapefactor_4]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error transforming data: {str(e)}")
 
    # Data preprocessing
    try:
        input_data = scaler.transform(input_data)
        input_data = pca.transform(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in data transformation using scaler/PCA: {str(e)}")

    # Model prediction
    try:
        prediction = svc.predict(input_data).item()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Return result
    return {
        "Predict": prediction,
        "Pistachio_type": "Kirmizi_Pistachio" if prediction == 0 else "Siirt_Pistachio"
    }


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/health')
async def service_health():
    """Service health status."""
    return {"OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)