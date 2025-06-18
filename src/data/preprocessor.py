import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from config import settings


class TextPreprocessor:
    """Preprocess text data for sentiment analysis."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = None
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download necessary NLTK data files."""
        for resource in settings.text.nltk_downloads:
            nltk.download(resource, quiet=True)

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the text."""
        if not isinstance(text, str):
            return ""

        text = text.casefold()
        text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters

        tokens = word_tokenize(text)  # Tokenize the text

        tokens = [  # Lemmatize and remove stop words
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        return " ".join(tokens)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """ "
        Process all text in dataframe and remove empty results.
        """

        if settings.verbose:
            print("Processing text data...")

        # Applying preprocessing
        df["processed_text"] = df["reviewText"].apply(self.preprocess_text)

        # Removing empty reviews after preprocessing
        df_clean = df[df["processed_text"].str.len() > 0].copy()

        if settings.verbose:
            print(f"Dataset shape after preprocessing: {df_clean.shape}")

        return df_clean

    def create_tfidf_vectorizer(self, df: pd.DataFrame) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with settings from config."""
        self.vectorizer = TfidfVectorizer(
            max_features=settings.text.max_features,
            stop_words="english",
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