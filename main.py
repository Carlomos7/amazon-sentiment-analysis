from config import settings
from src.data import DataLoader, TextPreprocessor
from src.training import (
    tune_model,
    get_models_and_params,
    evaluate_model,
    find_best_model,
)
from src.visualization import create_all_plots
from src.utils import split_data, print_dataset_info


def main():
    """Main pipeline execution."""
    print("=== Amazon Sentiment Analysis Pipeline ===")

    # 1. Data Loading
    print("\n1. Loading and preparing dataset...")
    loader = DataLoader()
    df = loader.load_and_prepare()
    print_dataset_info(df)

    # 2. Text Preprocessing
    print("\n2. Text preprocessing...")
    preprocessor = TextPreprocessor()
    df = preprocessor.process_dataframe(df)

    # 3. Train/Test Split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df["processed_text"], df["sentiment_label"]
    )

    if settings.verbose:
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

    # 4. TF-IDF Vectorization
    print("\n4. Creating TF-IDF features...")
    X_train_tfidf = preprocessor.fit_transform_tfidf(X_train)
    X_test_tfidf = preprocessor.transform_tfidf(X_test)

    # 5. Model Training with Hyperparameter Tuning
    print("\n5. Training and tuning models...")
    models_config = get_models_and_params()
    tuned_models = {}

    for name, config in models_config.items():
        tuned_models[name] = tune_model(
            config["model"], config["params"], name, X_train_tfidf, y_train
        )

    # 6. Model Evaluation
    print("\n6. Evaluating models...")
    all_metrics = []
    all_cms = []
    all_roc_data = []

    for name, model in tuned_models.items():
        metrics, fpr, tpr, cm = evaluate_model(model, X_test_tfidf, y_test, name)
        all_metrics.append(metrics)
        all_cms.append(cm)
        all_roc_data.append((fpr, tpr))

    # 7. Find Best Model
    best_model_name = find_best_model(all_metrics)

    # 8. Generate Visualizations
    print("\n7. Generating visualizations...")
    create_all_plots(df, all_metrics, all_cms, all_roc_data)

    print(f"\n=== Pipeline completed successfully! ===")
    print(f"Best performing model: {best_model_name}")
    print(f"Results saved to: {settings.plots_dir}")


if __name__ == "__main__":
    main()
