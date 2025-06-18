from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import List


class Settings(BaseSettings):
    """ "
    Main configuration
    """

    kaggle_dataset: str = "tarkkaanko/amazon"
    kaggle_file_path: str = "amazon_reviews.csv"

    test_size: float = 0.2
    random_state: int = 42

    max_features: int = 10000
    min_df: int = 5

    plots_dir: Path = Path("output/plots")

    verbose: bool = True

    class TextConfig(BaseModel):
        """Text preprocessing settings."""

        max_features: int = 10000
        min_df: int = 5
        nltk_downloads: List[str] = ["punkt", "stopwords", "wordnet"]

    text: TextConfig = TextConfig()

    def create_dirs(self):
        """Create output directories."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()
