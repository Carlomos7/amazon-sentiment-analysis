from typing import List
from pydantic import BaseModel


class ModelMetrics(BaseModel):
    """Model metrics for evaluation."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float


class TrainingResults(BaseModel):
    """Results of the training process."""

    all_metrics: List[ModelMetrics]
    best_model: str

    def get_best_model(self) -> ModelMetrics:
        """Get the metrics of the best model."""
        return max(self.all_metrics, key=lambda x: x.f1_score)
