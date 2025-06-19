import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from config import settings


class TextPreprocessor:
    """Handles text preprocessing and TF-IDF vectorization."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = None
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download required NLTK data."""
        for resource in settings.text.nltk_downloads:
            nltk.download(resource, quiet=True)

    def preprocess_text(self, text):
        """
        Clean and preprocess text data.
        Exact same logic as the original preprocess_text function.
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.casefold()

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove non-alphabetic characters and keep spaces
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        # Return preprocessed text
        return " ".join(tokens)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all text in dataframe and remove empty results.
        """
        if settings.verbose:
            print("Processing text...")

        # Apply preprocessing
        df["processed_text"] = df["reviewText"].apply(self.preprocess_text)

        # Remove empty reviews after preprocessing
        df_clean = df[df["processed_text"].str.len() > 0].copy()

        if settings.verbose:
            print(f"Dataset shape after preprocessing: {df_clean.shape}")

        return df_clean

    def create_tfidf_vectorizer(self):
        """Create TF-IDF vectorizer with settings from config."""
        self.vectorizer = TfidfVectorizer(
            max_features=settings.text.max_features, min_df=settings.text.min_df
        )
        return self.vectorizer

    def fit_transform_tfidf(self, texts):
        """Fit vectorizer and transform texts."""
        if self.vectorizer is None:
            self.create_tfidf_vectorizer()

        transformed = self.vectorizer.fit_transform(texts)

        if settings.verbose:
            print(f"Number of features: {transformed.shape[1]}")

        return transformed

    def transform_tfidf(self, texts):
        """Transform texts using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform_tfidf first.")

        return self.vectorizer.transform(texts)
