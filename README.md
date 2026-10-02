# Food Ordering Chatbot

A chatbot that takes takeaway orders in natural language, answers factual questions, and handles small talk. It is built from classical NLP components (TF-IDF, cosine similarity, logistic regression) and a finite state machine, with no large language model involved.

```
You: my name is Ron
Bot: Nice to meet you, Ron! I'll remember your name.
You: who invented pizza
Bot: Modern pizza originated in Naples, Italy.
You: what is on the menu
Bot: Here's our full menu:
- Margherita Pizza: small £6.49, medium £8.49, large £10.49
...
```

## How it works

Each message passes through three stages:

1. **Routing** (`nlp_router.py`). A hybrid router decides whether the message is a factual question or a task. Rule-based overrides catch the clear cases (ordering phrases, identity questions, small talk phrased as a question). Everything else falls through to a TF-IDF + logistic regression classifier, which only routes to question answering when it is at least 65% confident.
2. **Intent matching and retrieval** (`nlp_utils.py`). Intents are matched by cosine similarity against 244 example utterances across 32 intents, with a string-similarity fallback for very short messages. Factual questions are answered by retrieving the closest of 60 stored question/answer pairs.
3. **Dialogue management** (`chatbot.py`, `food_ordering.py`). An order is tracked by a finite state machine that moves through item, size, quantity, further items, delivery or pickup, address and confirmation. The user can ask a question or make small talk mid-order and the bot returns to the point the order had reached.

Other features:

- Several items in one message ("two large pepperoni pizzas and a cola"), asking in turn for any size that is missing
- Fuzzy matching of misspelt menu items, with suggestions
- Size validation per item, running totals and an order summary
- Remembers the user's name and can repeat its last message

## Running it

Requires Python 3.9+.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This opens the chat in your browser, with a side panel showing the order as it builds. For the plain command-line version:

```bash
python main.py
```

Type `exit` to quit. The router model is trained from `data/router_data.json` if `data/router_model.joblib` is missing.

## Evaluation

```bash
python test_model.py
```

This prints accuracy, a classification report and a confusion matrix for the router, the intent matcher and the question-answer retriever.

## Project layout

| File | Purpose |
|---|---|
| `streamlit_app.py` | Web chat interface |
| `main.py` | Command-line entry point |
| `chatbot.py` | Dialogue controller that ties the components together |
| `nlp_router.py` | Question vs. task routing (rules + logistic regression) |
| `nlp_utils.py` | Intent matcher and question-answer retriever |
| `food_ordering.py` | Menu data and the ordering state machine |
| `test_model.py` | Evaluation script |
| `data/` | Intent examples, question/answer pairs and router training data |

## Author

Ron Prekopuca
