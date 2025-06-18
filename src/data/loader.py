import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from config import settings


class DataLoader:

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from Kaggle."""

        if settings.verbose:
            print("Loading dataset from Kaggle...")

        # Testing the kagglehub pandas adapter
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            settings.kaggle_dataset_path,
            settings.kaggle_file_path,
        )

        if settings.verbose:
            print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")
        return df

    def filter_neutral_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove neutral reviews from the dataset."""

        if settings.verbose:
            print("Rating distribution before filtering:")
            print(df["overall"].value_counts().sort_index())

        # Remove neutral reviews (score of 3)
        df_filtered = df[df["overall"] != 3].copy()

        # Creating binary sentiment labels (0 for negative, 1 for positive)
        df_filtered["sentiment_label"] = df_filtered["overall"].apply(
            lambda x: 1 if x > 3 else 0
        )

        if settings.verbose:
            print(f"Dataset shape after filtering: {df_filtered.shape}")

        return df_filtered

    def load_and_filter(self) -> pd.DataFrame:
        """Load and filter the dataset."""
        df = self.load_dataset()
        df_filtered = self.filter_neutral_reviews(df)
        return df_filtered
