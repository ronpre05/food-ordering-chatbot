import json
import re
from pathlib import Path
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Router:
    """
    Hybrid router that classifies user input into:
    - 'qa'     : factual questions
    - 'intent' : task-oriented or transactional input
    """

    def __init__(
        self,
        data_path: str = "data/router_data.json",
        model_path: str = "data/router_model.joblib",
        retrain: bool = False,
    ):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)


        # Load or train ML router model

        if self.model_path.exists() and not retrain:
            self.vectorizer, self.model = load(self.model_path)
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = [d["text"] for d in data]
            labels = [d["label"] for d in data]

            self.vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
            X = self.vectorizer.fit_transform(texts)

            self.model = LogisticRegression(random_state=42, max_iter=600)
            self.model.fit(X, labels)

            dump((self.vectorizer, self.model), self.model_path)


        # Rule-based phrase libraries

        # Identity-related queries (always intent)
        self.IDENTITY_PHRASES = (
            "my name", "what's my name", "whats my name",
            "who am i", "remember my name", "did i tell you my name",
        )

        # System/help queries (always intent)
        self.SYSTEM_PHRASES = (
            "help", "what can you do", "commands", "how does this work", "features",
        )

        # Order verbs & food nouns
        self.ORDER_VERBS = (
            "order", "i want", "i'd like", "i would like", "i will have",
            "can i get", "can i have", "let me get", "give me",
            "add", "buy", "get me", "have",
        )

        self.FOOD_WORDS = (
            "pizza", "burger", "fries", "chips", "wings", "garlic bread",
            "cola", "coke", "drink", "juice", "dessert", "brownie", "cake",
        )

        # Order status phrases (must not route to QA)
        self.ORDER_STATUS_PHRASES = (
            "what did i order", "whats my order", "what’s my order",
            "what have i ordered", "show my order",
        )

        # Social WH-questions that should stay as intent
        self.SMALLTALK_WH_BLOCK = (
            "how are you", "how are u", "how’re you", "how’s it going",
            "are you ok", "how do you feel",
        )

        # WH-starters for QA detection
        self.WH_STARTERS = (
            "what", "when", "where", "who", "why", "how", "define", "explain",
        )


    # MAIN ROUTING METHOD

    def predict_route(self, text: str) -> str:
        """Predict the routing label for a user input."""

        t = text.lower().strip()

        
        # HARD OVERRIDES

        if any(p in t for p in self.IDENTITY_PHRASES):
            return "intent"

        if any(p in t for p in self.SYSTEM_PHRASES):
            return "intent"

        if any(p in t for p in self.ORDER_STATUS_PHRASES):
            return "intent"

        # smalltalk
        if any(p in t for p in self.SMALLTALK_WH_BLOCK):
            return "intent"

        # food-ordering should be intent
        if any(v in t for v in self.ORDER_VERBS) and any(
            w in t for w in self.FOOD_WORDS
        ):
            return "intent"

        if any(t == w for w in self.FOOD_WORDS):
            return "intent"

        
        # QA DETECTION BASED ON WH-STARTERS
        first_word = t.split(" ")[0] if t else ""

        if first_word in self.WH_STARTERS:

            # explicit QA about the chatbot itself
            if re.search(r"who (created|built|made) you", t):
                return "qa"

            # food-related factual questions
            if re.search(r"\b(invented|origin|history|calories|definition|what is)\b", t):
                return "qa"

            # block transactional questions about ordering
            if any(w in t for w in self.FOOD_WORDS) and any(
                v in t for v in self.ORDER_VERBS
            ):
                return "intent"

            if any(v in t for v in self.ORDER_VERBS):
                return "intent"

            # otherwise treat as factual QA
            return "qa"


        # MACHINE LEARNING FALLBACK

        X = self.vectorizer.transform([t])
        probs = self.model.predict_proba(X)[0]
        labels = self.model.classes_

        label = labels[probs.argmax()]
        confidence = probs.max()

        # slightly relaxed QA threshold to recover borderline QA
        if label == "qa" and confidence >= 0.65:
            return "qa"


        # SAFE DEFAULT

        return "intent"
