import json
import random
import re
from typing import Optional

from food_ordering import FoodOrderingSystem
from nlp_utils import IntentMatcher, QARetriever
from nlp_router import Router


# Keyword Lists

ORDER_TRIGGERS = [
    "order", "i want", "i'd like", "i want to order", "can i get", "can i have",
    "get me", "i'll have", "i will have", "add", "deliver", "delivery", "pickup"
]

FOOD_KEYWORDS = [
    "pizza", "burger", "fries", "chips", "wings", "garlic",
    "bread", "drink", "dessert", "cola", "ice"
]


class Chatbot:
    """
    Main dialogue controller that integrates:
    - Router 
    - Intent classification
    - QA retrieval
    - Food ordering FSM
    """

    def __init__(self, intents_path: str = "data/intents.json", qa_path: str = "data/qa_data.json"):
        self.memory = {}
        self.order_system = FoodOrderingSystem()

        # Load intents
        with open(intents_path, "r", encoding="utf-8") as f:
            self.intents = json.load(f)
        self.matcher = IntentMatcher(self.intents)

        # Load QA data
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        self.qa = QARetriever(qa_pairs)

        # Load router
        self.router = Router(retrain=False)

        print("(Chatbot initialised: Router + IntentMatcher + QA Retriever ready.)")

    
    # MAIN RESPONSE METHOD

    def respond(self, user_input: str) -> str:
        """Generate a response to the user input."""

        if not user_input:
            return "Please say something so I can respond!"

        text = user_input.strip()
        text_lower = text.lower()
        self.memory["last_user_message"] = text

        # NAME HANDLING 
        name_reply = self._handle_name_statements(text)
        if name_reply:
            self.memory["last_bot_message"] = name_reply
            return name_reply

        # REPEAT LAST BOT MESSAGE 
        if re.search(r"\b(remind me|repeat that|say that again|can you repeat that)\b", text_lower):
            return self.memory.get("last_bot_message", "I don't have anything to repeat.")

        # ROUTING & INTENT
        route = self.router.predict_route(text)
        intent = self.matcher.predict_intent(text)

        # MENU QUERY (ONLY WHEN IDLE)
        if intent == "menu_query" and self.order_system.state == "idle":
            menu_text = self.order_system.get_menu_text()
            self.memory["last_bot_message"] = menu_text
            return menu_text

        # ACTIVE ORDER HANDLING

        if self.order_system.state != "idle":

            # STRICT ADDRESS HANDLING
            if self.order_system.state == "awaiting_address" and self.order_system._looks_like_address(text):
                fsm_response = self.order_system.handle_intent("order_address", text)
                if fsm_response:
                    hint = self.order_system.get_state_hint()
                    if hint:
                        fsm_response += f"\n\n👉 {hint}"
                    self.memory["last_bot_message"] = fsm_response
                    return fsm_response

            # Allow QA during an active order
            if route == "qa":
                answer = self.qa.get_answer(text)
                if answer:
                    reply = answer + "\n\n(You can continue your order anytime!)"
                    self.memory["last_bot_message"] = reply
                    return reply

            # Help intent during active order
            if intent == "help":
                reply = self._render_intent_response("help") + "\n\n(You can continue your order anytime!)"
                self.memory["last_bot_message"] = reply
                return reply

            

            # If we're mid-order and no intent was recognised, still let the FSM try
            if not intent:
                fsm_response = self.order_system.handle_intent("", text)
                if fsm_response:
                    hint = self.order_system.get_state_hint()
                    if hint:
                        fsm_response += f"\n\n👉 {hint}"
                    self.memory["last_bot_message"] = fsm_response
                    return fsm_response


            # Let the FSM handle any other intent first 
            if intent:
                fsm_response = self.order_system.handle_intent(intent, text)
                if fsm_response:
                    hint = self.order_system.get_state_hint()
                    if hint:
                        fsm_response += f"\n\n👉 {hint}"
                    self.memory["last_bot_message"] = fsm_response
                    return fsm_response

            # If FSM didn’t handle it and it’s non-order use canned reply
            if intent and not intent.startswith("order_"):
                intent_reply = self._render_intent_response(intent)
                if intent_reply:
                    reply = (
                        intent_reply
                        + f"\n\n🛒 Current order items: {len(self.order_system.order['items'])}"
                        + "\n👉 You can continue your order anytime!"
                    )
                    self.memory["last_bot_message"] = reply
                    return reply
                

        # CLEAN QA (WHEN NOT ORDERING)
        if route == "qa":
            answer = self.qa.get_answer(text)
            if answer:
                self.memory["last_bot_message"] = answer
                return answer

            fallback = "I'm not entirely sure, but I can check that next time!"
            self.memory["last_bot_message"] = fallback
            return fallback


        # HEURISTIC ORDER DETECTION 
        if self._looks_like_order_request(text_lower):
            fsm_response = self.order_system.handle_intent(intent or "order_item", text)
            if fsm_response:
                hint = self.order_system.get_state_hint()
                if hint:
                    fsm_response += f"\n\n👉 {hint}"
                self.memory["last_bot_message"] = fsm_response
                return fsm_response


        # IDENTITY GET 
        if intent == "identity_get":
            name = self.memory.get("name")
            reply = f"Your name is {name}." if name else "I don't know your name yet."
            self.memory["last_bot_message"] = reply
            return reply

        # SMALLTALK 
        if intent and not intent.startswith("order_"):
            intent_reply = self._render_intent_response(intent)
            if intent_reply:
                if intent.startswith("smalltalk_") and random.random() < 0.4:
                    intent_reply += " 😊"
                self.memory["last_bot_message"] = intent_reply
                return intent_reply

        if intent in ("smalltalk_confirmation", "smalltalk_thanks"):
            hint = self.order_system.get_state_hint()
            reply = f"{self._render_intent_response(intent)}\n\n👉 {hint}"
            self.memory["last_bot_message"] = reply
            return reply

        # FALLBACK 
        error = "Sorry, I didn't understand that. Try asking for the menu or say 'I want to order food'."
        self.memory["last_bot_message"] = error
        return error

    
    # HELPER METHODS

    def _handle_name_statements(self, text: str) -> Optional[str]:
        pattern = re.search(r"(?:my name is|call me|i am)\s+([A-Za-z]+)", text, re.IGNORECASE)
        if pattern:
            name = pattern.group(1).capitalize()
            self.memory["name"] = name
            return f"Nice to meet you, {name}! I'll remember your name."
        return None

    def _looks_like_order_request(self, text_lower: str) -> bool:
        return any(trigger in text_lower for trigger in ORDER_TRIGGERS) or \
               any(word in text_lower for word in FOOD_KEYWORDS)

    def _render_intent_response(self, intent_name: Optional[str]) -> Optional[str]:
        if not intent_name:
            return None

        for item in self.intents:
            if item["name"] == intent_name:
                template = random.choice(item.get("responses", [""]))
                return template.format(
                    name=self.memory.get("name", "friend"),
                    last_bot_message=self.memory.get("last_bot_message", "")
                )
        return None
