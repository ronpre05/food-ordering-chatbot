import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class IntentMatcher:
    """
    Intent classification using TF-IDF vectorisation and cosine similarity.

    - Trained on example utterances from the intents JSON
    - Returns the most similar intent if the similarity exceeds a threshold
    - Includes a fallback string-similarity method for very short inputs
    """

    def __init__(self, intents, threshold: float = 0.35):
        self.intents = intents
        self.threshold = threshold

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)

        # Flatten training examples
        self.examples = [
            ex for intent in intents for ex in intent.get("examples", [])
        ]
        self.intent_lookup = [
            intent["name"] for intent in intents for _ in intent.get("examples", [])
        ]

        self.tfidf = (
            self.vectorizer.fit_transform(self.examples)
            if self.examples else None
        )

    
    # MAIN INTENT PREDICTION METHOD

    def predict_intent(self, text: str):
        

        text = text.lower().strip()

        
        # Fallback for very short messages

        if len(text.split()) <= 2:
            best_ratio = 0
            best_intent = None

            for i, ex in enumerate(self.examples):
                ratio = SequenceMatcher(None, text, ex.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_intent = self.intent_lookup[i]

            if best_ratio > 0.55:
                return best_intent


        # TF-IDF + Cosine Similarity Route

        if self.tfidf is None:
            return None

        sims = cosine_similarity(
            self.vectorizer.transform([text]), self.tfidf
        )[0]

        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score < self.threshold:
            return None

        return self.intent_lookup[best_idx]


class QARetriever:
    """
    TF-IDF based information retrieval system for factual question answering.

    - Uses cosine similarity over TF-IDF vectors
    - The router is the only authority that decides when QA is allowed
    - Returns the best-matching answer if above a similarity threshold
    """

    def __init__(self, qa_pairs, threshold: float = 0.35):
        self.threshold = threshold
        self.questions = [q["question"] for q in qa_pairs]
        self.answers = [q["answer"] for q in qa_pairs]

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
        self.tfidf = (
            self.vectorizer.fit_transform(self.questions)
            if self.questions else None
        )


    # MAIN QA RETRIEVAL METHOD

    def get_answer(self, user_input: str):

        if self.tfidf is None:
            return None

        sims = cosine_similarity(
            self.vectorizer.transform([user_input.lower().strip()]),
            self.tfidf,
        )[0]

        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score >= self.threshold:
            return self.answers[best_idx]

        return None
