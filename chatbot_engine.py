
import re
import numpy as np
import pandas as pd
import nltk
import streamlit as st

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


# ============================================================
# NLTK RESOURCES
# ============================================================

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words("english"))


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [
        token for token in tokens
        if token not in english_stopwords and len(token) > 1
    ]

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


# ============================================================
# MEDQUAD CHATBOT ENGINE
# ============================================================

class MedicalChatbotEngineV3:

    def __init__(self, dataframe, threshold=0.35, top_k=3):

        self.df = dataframe.copy().reset_index(drop=True)
        self.threshold = threshold
        self.top_k = top_k
        self.conversation_history = []

        required = ["question", "answer"]

        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.df = self.df.dropna(
            subset=["question", "answer"]
        ).reset_index(drop=True)

        self.df["question"] = self.df["question"].astype(str).str.strip()
        self.df["answer"] = self.df["answer"].astype(str).str.strip()

        self.df = self.df[
            (self.df["question"] != "") &
            (self.df["answer"] != "")
        ].reset_index(drop=True)

        # Build search text from original question and category
        if "category" in self.df.columns:
            self.df["search_text"] = (
                self.df["category"].fillna("").astype(str) + ". " +
                self.df["question"]
            )
        else:
            self.df["search_text"] = self.df["question"]

        # Preprocess the original questions consistently
        self.df["processed_search"] = (
            self.df["search_text"].apply(preprocess_text)
        )

        self._build_index()
        self._define_rules()

    # ========================================================
    # BUILD SEARCH INDEX
    # ========================================================

    def _build_index(self):

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["processed_search"]
        )

        # Semantic model
        self.semantic_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.semantic_embeddings = self.semantic_model.encode(
            self.df["search_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        print("Search index berhasil dibuat.")
        print("Jumlah FAQ:", len(self.df))

    # ========================================================
    # RULES
    # ========================================================

    def _define_rules(self):

        self.greeting_patterns = [
            r"\bhi\b",
            r"\bhello\b",
            r"\bhey\b",
            r"\bgood morning\b",
            r"\bgood afternoon\b",
            r"\bgood evening\b",
        ]

        self.emergency_patterns = [
            r"\bsevere chest pain\b",
            r"\bchest pain\b",
            r"\bdifficulty breathing\b",
            r"\bshortness of breath\b",
            r"\bcan't breathe\b",
            r"\bcannot breathe\b",
            r"\bunconscious\b",
            r"\bsevere bleeding\b",
            r"\bstroke symptoms\b",
            r"\boverdose\b",
            r"\bsesak napas\b",
            r"\bnyeri dada hebat\b",
            r"\bpendarahan hebat\b",
        ]

    def _check_rules(self, user_input):

        text = user_input.lower().strip()

        for pattern in self.emergency_patterns:
            if re.search(pattern, text):
                return "emergency"

        if text in ["hi", "hello", "hey", "good morning",
                    "good afternoon", "good evening"]:
            return "greeting"

        return None

    # ========================================================
    # SEARCH TF-IDF
    # ========================================================

    def _search_tfidf(self, query):

        processed_query = preprocess_text(query)

        if not processed_query.strip():
            return np.zeros(len(self.df))

        query_vector = self.vectorizer.transform([processed_query])

        return cosine_similarity(
            query_vector, self.tfidf_matrix
        ).flatten()

    # ========================================================
    # SEARCH SEMANTIC
    # ========================================================

    def _search_semantic(self, query):

        query_embedding = self.semantic_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = np.dot(
            self.semantic_embeddings,
            query_embedding[0]
        )

        return scores

    # ========================================================
    # FIND BEST MATCH
    # ========================================================

    def _find_best_match(self, query):

        tfidf_scores = self._search_tfidf(query)
        semantic_scores = self._search_semantic(query)

        # Normalize TF-IDF to 0-1 range for combining
        if tfidf_scores.max() > 0:
            tfidf_normalized = tfidf_scores / tfidf_scores.max()
        else:
            tfidf_normalized = tfidf_scores

        # Semantic similarity has more weight
        combined_scores = (
            0.35 * tfidf_normalized +
            0.65 * semantic_scores
        )

        top_indices = np.argsort(combined_scores)[::-1][:self.top_k]

        results = self.df.iloc[top_indices].copy()

        results["tfidf_score"] = tfidf_scores[top_indices]
        results["semantic_score"] = semantic_scores[top_indices]
        results["similarity_score"] = combined_scores[top_indices]

        return results.reset_index(drop=True)

    # ========================================================
    # RESPONSE
    # ========================================================

    def get_response(self, user_input):

        user_input = str(user_input).strip()

        if not user_input:
            return "Please enter a medical question."

        rule_result = self._check_rules(user_input)

        if rule_result == "emergency":
            return (
                "Your message may describe a medical emergency. "
                "Please contact local emergency services or go "
                "to the nearest emergency department now. "
                "Do not rely on this chatbot for emergency care."
            )

        if rule_result == "greeting":
            return (
                "Hello! You can ask me a medical question "
                "about health."
            )

        # Search only the current question.
        # Do not append previous questions automatically.
        results = self._find_best_match(user_input)

        if results.empty:
            return "No relevant FAQ was found in the dataset."

        best_result = results.iloc[0]
        best_score = float(best_result["similarity_score"])

        print("\n========== CHATBOT DEBUG ==========")
        print("User input:", user_input)
        print(
            results[
                ["question", "tfidf_score",
                 "semantic_score", "similarity_score"]
            ].to_string(index=False)
        )
        print("===================================\n")

        if best_score < self.threshold:
            response = (
                "I could not find a sufficiently relevant "
                "answer in my FAQ database. Please consult "
                "a healthcare professional."
            )
        else:
            category = best_result.get("category", "Not specified")

            response = (
                f"Category: {category}\n\n"
                f"Question: {best_result['question']}\n\n"
                f"Answer: {best_result['answer']}\n\n"
                "Note: This information is educational "
                "and is not a medical diagnosis."
            )

        self.conversation_history.append(user_input)
        self.conversation_history = self.conversation_history[-5:]

        return response

    # ========================================================
    # RESET CONVERSATION
    # ========================================================

    def reset_conversation(self):
        self.conversation_history = []
        return "Conversation history has been reset."
