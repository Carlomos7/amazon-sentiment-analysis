import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import settings


def plot_sentiment_distribution(df):
    """Plot sentiment distribution"""
    plt.figure(figsize=(10, 6))
    sentiment_counts = df["sentiment_label"].value_counts().sort_index()
    ax = sns.barplot(
        x=["Negative", "Positive"],
        y=sentiment_counts.values,
        palette=["firebrick", "forestgreen"],
    )

    # Adding count labels on the bars
    for i, count in enumerate(sentiment_counts.values):
        ax.text(
            i,
            count / 2,
            f"{count}\n({count/len(df):.1%})",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    plt.title("Sentiment Distribution in Amazon Reviews", fontsize=15)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Review Count", fontsize=12)
    plt.tight_layout()

    plot_path = settings.plots_dir / "sentiment_distribution.png"
    plt.savefig(plot_path)
    plt.close()

    if settings.verbose:
        print(f"Saved sentiment distribution plot to {plot_path}")


def plot_confusion_matrices(all_metrics, all_cms):
    """Plot combined confusion matrices"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [metric.model_name for metric in all_metrics]

    for i, (title, cm) in enumerate(zip(titles, all_cms)):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=axes[i],
        )
        axes[i].set_title(f"{title} Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()

    plot_path = settings.plots_dir / "combined_confusion_matrices.png"
    plt.savefig(plot_path)
    plt.close()

    if settings.verbose:
        print(f"Saved confusion matrices to {plot_path}")


def plot_model_comparison(all_metrics):
    """Plot model performance comparison"""
    plt.figure(figsize=(12, 6))
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    x = np.arange(len(metrics))
    width = 0.25

    colors = ["darkorange", "blue", "green"]

    # Plot bars for each model
    for i, (metric_obj, color) in enumerate(zip(all_metrics, colors)):
        values = [getattr(metric_obj, m) for m in metrics]
        bars = plt.bar(
            x + i * width - width,
            values,
            width,
            label=metric_obj.model_name,
            color=color,
        )

        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                color=color,
                fontweight="bold",
            )

    # Labels and formatting
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Performance Comparison", fontsize=15)
    plt.xticks(x, [m.replace("_", " ").title() for m in metrics])
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()

    plot_path = settings.plots_dir / "model_performance_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    if settings.verbose:
        print(f"Saved performance comparison to {plot_path}")


def plot_roc_curves(all_metrics, all_roc_data):
    """Plot combined ROC curves"""
    plt.figure(figsize=(8, 6))
    colors = ["darkorange", "blue", "green"]

    for metric_obj, (fpr, tpr), color in zip(all_metrics, all_roc_data, colors):
        plt.plot(
            fpr,
            tpr,
            label=f"{metric_obj.model_name} (AUC = {metric_obj.auc_score:.2f})",
            color=color,
            lw=2,
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plot_path = settings.plots_dir / "combined_roc_curve.png"
    plt.savefig(plot_path)
    plt.close()

    if settings.verbose:
        print(f"Saved ROC curves to {plot_path}")


def create_all_plots(df, all_metrics, all_cms, all_roc_data):
    """Create all visualization plots."""
    # Ensure output directory exists
    settings.create_dirs()

    plot_sentiment_distribution(df)
    plot_confusion_matrices(all_metrics, all_cms)
    plot_model_comparison(all_metrics)
    plot_roc_curves(all_metrics, all_roc_data)

    if settings.verbose:
        print("\nAll visualizations completed successfully!")
