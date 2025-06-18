from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """ "
    Main configuration
    """

    kaggle_dataset: str = "tarkkaanko/amazon"
    kaggle_file_path: str = ""

    test_size: float = 0.2
    random_state: int = 42

    max_features: int = 10000
    min_df: int = 5

    plot_dir: Path = Path("output/plots")
    models_dir: Path = Path("output/models")

    verbose: bool = True

    def create_dirs(self):
        """Create output directories."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()