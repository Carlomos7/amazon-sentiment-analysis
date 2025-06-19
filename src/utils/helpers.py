from sklearn.model_selection import train_test_split
from config import settings


def split_data(X, y):
    """Split data into train/test sets with consistent parameters."""
    return train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )


def print_dataset_info(df):
    """Print dataset information in consistent format."""
    if not settings.verbose:
        return

    total = len(df)
    positive = len(df[df["sentiment_label"] == 1])
    negative = len(df[df["sentiment_label"] == 0])

    print(f"\n=== Dataset Information ===")
    print(f"Total reviews: {total:,}")
    print(f"Positive reviews: {positive:,} ({positive/total:.1%})")
    print(f"Negative reviews: {negative:,} ({negative/total:.1%})")
