from pydantic import BaseModel, Field


class BreastCancerInput(BaseModel):
    mean_radius: float = Field(..., gt=0)
    mean_concavity: float = Field(..., gt=0)
    mean_concave_points: float = Field(..., gt=0)
    radius_error: float = Field(..., gt=0)
    worst_perimeter: float = Field(..., gt=0)
    worst_concavity: float = Field(..., gt=0)
    worst_concave_points: float = Field(..., gt=0)
    worst_symmetry: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class BatchPredictionRequest(BaseModel):
    samples: list[BreastCancerInput]
