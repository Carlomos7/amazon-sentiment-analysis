from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from src.models import ModelMetrics
from config import settings


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the trained model using various metrics."""
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract metrics for positive class
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    if settings.verbose:
        print(f"\n=== {model_name} Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc_score:.4f}")

    return (
        ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc_score,
        ),
        fpr,
        tpr,
        cm,
    )


def find_best_model(all_metrics):
    """Find the best model based on F1 score."""
    best_metrics = max(all_metrics, key=lambda x: x.f1_score)

    if settings.verbose:
        print(f"\n=== Results Summary ===")
        print(f"Best model based on F1 score: {best_metrics.model_name}")
        print("Performance comparison:")
        for metric in all_metrics:
            print(
                f"  {metric.model_name}: Accuracy={metric.accuracy:.4f}, "
                f"F1={metric.f1_score:.4f}, AUC={metric.auc_score:.4f}"
            )

    return best_metrics.model_name
