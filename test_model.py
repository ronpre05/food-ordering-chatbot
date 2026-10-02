import json
import random
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from nlp_router import Router
from nlp_utils import IntentMatcher, QARetriever

"""
EVALUATION SCRIPT 

This script evaluates:
1. Router (qa | intent | chitchat)
2. IntentMatcher 
3. QARetriever 

reports:
- Accuracy
- Precision, Recall, F1-score 
- Confusion matrices
"""


# ROUTER EVALUATION

def evaluate_router(data_path: str = "data/router_data.json"):
    print("\n=== ROUTER EVALUATION ===")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    true_labels = [d["label"] for d in data]

    # Shuffle to avoid ordering bias
    combined = list(zip(texts, true_labels))
    random.shuffle(combined)
    texts, true_labels = zip(*combined)

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, true_labels, test_size=0.3, random_state=42
    )

    # Train router
    router = Router(retrain=True)

    # Predict
    y_pred = [router.predict_route(t) for t in X_test]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



# INTENT MATCHER EVALUATION

def evaluate_intents(intents_path: str = "data/intents.json"):
    print("\n=== INTENT MATCHER EVALUATION ===")

    with open(intents_path, "r", encoding="utf-8") as f:
        intents = json.load(f)

    texts = []
    labels = []

    for intent in intents:
        for ex in intent.get("examples", []):
            texts.append(ex)
            labels.append(intent["name"])

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    matcher = IntentMatcher(intents)

    y_pred = [matcher.predict_intent(x) for x in X_test]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



# QA RETRIEVER EVALUATION

def evaluate_qa(qa_path: str = "data/qa_data.json"):
    print("\n=== QA RETRIEVER EVALUATION ===")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    questions = [q["question"] for q in qa_pairs]
    answers = [q["answer"] for q in qa_pairs]

    X_train, X_test, y_train, y_test = train_test_split(
        questions, answers, test_size=0.3, random_state=42
    )

    qa = QARetriever(qa_pairs)

    correct = 0
    for q, true_a in zip(X_test, y_test):
        pred = qa.get_answer(q)
        if pred == true_a:
            correct += 1

    accuracy = correct / len(X_test)
    print("QA Retrieval Accuracy:", accuracy)



# MAIN

if __name__ == "__main__":
    evaluate_router()
    evaluate_intents()
    evaluate_qa()
