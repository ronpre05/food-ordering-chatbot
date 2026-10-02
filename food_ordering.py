import re
from copy import deepcopy
from difflib import get_close_matches


# MENU DATA 

MENU_ITEMS = {
    "margherita pizza": {"sizes": {"small": 6.49, "medium": 8.49, "large": 10.49}},
    "pepperoni pizza": {"sizes": {"small": 6.99, "medium": 8.99, "large": 11.49}},
    "bbq chicken pizza": {"sizes": {"small": 7.49, "medium": 9.49, "large": 11.99}},
    "cheeseburger": {"sizes": {"single": 4.49, "double": 6.49}},
    "chicken burger": {"sizes": {"single": 4.99, "double": 6.99}},
    "veggie burger": {"sizes": {"single": 4.29, "double": 6.29}},
    "fries": {"sizes": {"small": 1.99, "medium": 2.49, "large": 2.99}},
    "onion rings": {"sizes": {"small": 2.29, "large": 3.99}},
    "wings": {"sizes": {"6-piece": 4.99, "10-piece": 7.99}},
    "garlic bread": {"sizes": {"regular": 2.49, "cheesy": 3.49}},
    "cola": {"sizes": {"can": 1.29, "bottle": 1.99}},
    "lemonade": {"sizes": {"can": 1.29, "bottle": 1.99}},
    "orange juice": {"sizes": {"small": 1.79, "large": 2.49}},
    "ice cream": {"sizes": {"single scoop": 1.49, "double scoop": 2.49}},
    "chocolate fudge cake": {"sizes": {"slice": 2.99}},
    "brownie": {"sizes": {"single": 1.99, "box": 5.49}},
}

NUM_WORDS = {
    "one": 1, "1": 1, "single": 1,
    "two": 2, "2": 2,
    "three": 3, "3": 3,
    "four": 4, "4": 4,
    "a": 1, "an": 1,
}


class FoodOrderingSystem:
    """
    Finite State Machine (FSM) for managing food orders.

    States:
    - idle
    - awaiting_item
    - awaiting_size
    - awaiting_quantity
    - awaiting_additional
    - awaiting_method
    - awaiting_address
    - awaiting_confirmation
    """

    def __init__(self):
        self.reset()
        self.last_order = None
        self.last_order_struct = None

    # ORDER RESET & INITIALISATION

    def reset(self):
        self.state = "idle"
        self.order = {"items": [], "method": None, "address": None}
        self.temp_item = None
        self.temp_size = None
        self.temp_quantity = None

    def start_order(self):
        self.state = "awaiting_item"
        return "Sure — what would you like to order first?"


    # MAIN FSM HANDLER

    def handle_intent(self, intent: str, user_input: str):
        text = (user_input or "").strip()
        text_lower = text.lower()

        # GLOBAL ACTIONS

        if intent == "order_cancel" or re.search(r"\b(cancel|nevermind|forget it|stop order|stop)\b", text_lower):
            self.reset()
            return "Order cancelled. Let me know if you need anything else!"

        if intent == "conversation_control_restart" or re.search(r"\b(start over|restart|reset order|clear everything)\b", text_lower):
            self.reset()
            return "Okay — starting over. What would you like to do?"

        if intent == "order_status" or re.search(r"\b(what did i order|what's my order|show my order|what's in my cart|what have i ordered)\b", text_lower):
            return self.get_last_order_summary()

        # ADD MORE ITEMS 

        if intent in {"order_add_more", "order_add_existing"} or (
            re.search(r"\b(add|also|another|more|and)\b", text_lower)
            and self._contains_item_word(text_lower)
        ):
            self.state = "awaiting_item"
            if self._contains_item_word(text_lower):
                return self._handle_item_input(text_lower)
            return "Sure — what would you like to add?"

        # IDLE STATE 

        if self.state == "idle":
            if intent == "order_start":
                return self.start_order()

            if intent == "order_item" or self._contains_item_word(text_lower):
                self.state = "awaiting_item"
                return self._handle_item_input(text_lower)

            if intent == "order_address" or re.search(r"\b(deliver to|address is)\b", text_lower):
                addr = self._extract_address(text)
                if addr:
                    self.order["address"] = addr
                    return f"Got it — I'll deliver to: {addr}. Would you like to confirm your order?"
                return "Please tell me the address you'd like delivery to."

            return None

        # FSM STATES 

        if self.state == "awaiting_item":
            if intent == "order_item" or self._contains_item_word(text_lower):
                return self._handle_item_input(text_lower)
            return "Sorry we don't have that item — try ask for the menu."

        if self.state == "awaiting_size":
            if intent == "order_size" or self._looks_like_size(text_lower):
                size = self._extract_size(text_lower, item=self.temp_item)
                if not size:
                    sizes = self._allowed_sizes_for(self.temp_item)
                    return f"Please choose a size. We have: {', '.join(sizes)}."
                self.temp_size = size
                self.state = "awaiting_quantity"
                return "How many would you like?"

            if re.search(r"\b(what sizes|what size|which sizes)\b", text_lower):
                sizes = self._allowed_sizes_for(self.temp_item)
                return f"We have: {', '.join(sizes)}."

            return "Please tell me a size (e.g. small, medium, large)."

        if self.state == "awaiting_quantity":
            if intent == "order_quantity" or self._looks_like_quantity(text_lower):
                qty = self._extract_quantity(text_lower)
                self.temp_quantity = qty or 1

                if not self.temp_item:
                    return "Which item would you like?"
                if not self.temp_size:
                    return "Which size would you like?"

                self.order["items"].append({
                    "item": self.temp_item,
                    "size": self.temp_size,
                    "quantity": self.temp_quantity,
                })

                self.temp_item = None
                self.temp_size = None
                self.temp_quantity = None
                self.state = "awaiting_additional"
                return "Added to your order. Would you like anything else?"

            return "Tell me how many you'd like (e.g. 1 or 2)."

        if self.state == "awaiting_additional":
            if intent in {"order_add_more", "order_item", "order_add_existing"} or self._contains_item_word(text_lower):
                self.state = "awaiting_item"
                return self._handle_item_input(text_lower) if self._contains_item_word(text_lower) else "Sure — what else would you like?"

            if intent == "order_method" or re.search(r"\b(delivery|deliver|pickup|pick up|collect)\b", text_lower):
                chosen = "delivery" if "deliver" in text_lower else "pickup"
                self.order["method"] = chosen

                if chosen == "delivery":
                    self.state = "awaiting_address"
                    addr = self._extract_address(text)
                    if addr:
                        self.order["address"] = addr
                        self.state = "awaiting_confirmation"
                        return self._summarise_order() + "\n\nWould you like to confirm your order?"
                    return "Delivery selected. Please tell me the delivery address."

                self.state = "awaiting_confirmation"
                return self._summarise_order() + "\n\nWould you like to confirm your order?"

            if intent == "order_confirm":
                self.state = "awaiting_confirmation"
                return self._summarise_order() + "\n\nShall I place the order?"

            if re.search(r"\b(no|nope|nah|that's all|done|no thanks)\b", text_lower):
                self.state = "awaiting_method"
                return "Delivery or pickup?"

            return "You can add more items, choose delivery/pickup, or confirm your order."

        if self.state == "awaiting_method":
            if intent == "order_method" or re.search(r"\b(delivery|deliver|pickup|pick up|collect)\b", text_lower):
                chosen = "delivery" if "deliver" in text_lower else "pickup"
                self.order["method"] = chosen

                if chosen == "delivery":
                    self.state = "awaiting_address"
                    addr = self._extract_address(text)
                    if addr:
                        self.order["address"] = addr
                        self.state = "awaiting_confirmation"
                        return self._summarise_order() + "\n\nWould you like to confirm your order?"
                    return "Please tell me the delivery address."

                self.state = "awaiting_confirmation"
                return self._summarise_order() + "\n\nWould you like to confirm your order?"

            return "Please say 'delivery' or 'pickup'."

        if self.state == "awaiting_address":
            
            text = text or ""
            cleaned = text.strip()

            if cleaned:
                addr = self._extract_address(text) or cleaned
                self.order["address"] = addr
                self.state = "awaiting_confirmation"
                return (
                    f"Got it — I'll deliver to: {addr}.\n\n"
                    f"{self._summarise_order()}\n\n"
                    "Would you like to confirm your order?"
                )

            return "Please provide the delivery address (e.g. 'deliver to 12 High Road')."


        if self.state == "awaiting_confirmation":
            if intent == "order_confirm":
                confirmation = self.last_order_summary_snapshot()
                self.last_order = confirmation
                self.last_order_struct = deepcopy(self.order)
                self.state = "idle"
                return confirmation

            if intent in {"order_add_more", "order_add_existing"} or self._contains_item_word(text_lower):
                self.state = "awaiting_item"
                return self._handle_item_input(text_lower) if self._contains_item_word(text_lower) else "Okay — what would you like to add?"

            if re.search(r"\b(cancel|nevermind|stop)\b", text_lower):
                self.reset()
                return "Order cancelled. Let me know if you need anything else!"

            if re.search(r"\b(no|nope|nah)\b", text_lower):
                self.state = "awaiting_additional"
                return "Okay — what would you like to add?"

            return "Say 'yes' to confirm, or 'no' to add more or cancel."

        return None

    
    # ITEM HANDLING HELPERS

    def _handle_item_input(self, text_lower: str):
        qty = self._extract_quantity(text_lower)
        item = self._find_item_in_text(text_lower)
        size = self._extract_size(text_lower, item=item)

        if item and item not in MENU_ITEMS:
            suggestion = self._suggest_item(item)
            if suggestion:
                return f"Sorry we don't have '{item}'. Did you mean '{suggestion}'?"
            return "Sorry, we don't sell that item. You can ask for the menu to see available items."

        if item and size and qty:
            if size not in MENU_ITEMS[item]["sizes"]:
                allowed = ", ".join(MENU_ITEMS[item]["sizes"].keys())
                return f"Sorry that size isn't available for {item}. Available sizes: {allowed}."

            self.order["items"].append({
                "item": item,
                "size": size,
                "quantity": qty,
            })
            self.state = "awaiting_additional"
            return f"Added {qty} × {size} {item}. Would you like anything else?"

        if item and size:
            self.temp_item = item
            self.temp_size = size
            self.state = "awaiting_quantity"
            return "How many would you like?"

        if item:
            self.temp_item = item
            self.state = "awaiting_size"
            return f"Great — what size {item} would you like?"

        return "I didn't catch the item. What would you like to order?"

 
    # NLP EXTRACTION UTILITIES

    def _find_item_in_text(self, text_lower: str):
        drink_match = re.search(r"(can|bottle)\s+(?:of\s+)?(coke|cola|coca cola|coca-cola)", text_lower)
        if drink_match:
            return "cola"

        for name in sorted(MENU_ITEMS.keys(), key=lambda x: -len(x)):
            if re.search(r"\b" + re.escape(name) + r"\b", text_lower):
                return name

        # fuzzy match 
        names = list(MENU_ITEMS.keys())
        close = get_close_matches(text_lower, names, n=1, cutoff=0.6)
        if close:
            return close[0]

        # category mapping as a *last resort*
        tokens = re.findall(r"\w+", text_lower)
        cat_map = {
            "pizza": "margherita pizza",
            "burger": "cheeseburger",
            "fries": "fries",
            "chips": "fries",
            "wings": "wings",
            "garlic": "garlic bread",
            "cola": "cola",
            "coke": "cola",
            "drink": "cola",
            "ice": "ice cream",
            "brownie": "brownie",
            "cake": "chocolate fudge cake",
        }

        for t in tokens:
            if t in cat_map:
                return cat_map[t]

        return None

    def _suggest_item(self, item_name: str):
        names = list(MENU_ITEMS.keys())
        close = get_close_matches(item_name, names, n=1, cutoff=0.5)
        return close[0] if close else None

    def _contains_item_word(self, text_lower: str) -> bool:
        keywords = {"pizza", "burger", "fries", "chips", "wings", "onion", "garlic", "bread", "drink", "dessert", "cola", "ice"}
        return any(k in text_lower for k in keywords)

    def _looks_like_size(self, text_lower: str) -> bool:
        sizes = set()
        for v in MENU_ITEMS.values():
            sizes.update(v["sizes"].keys())
        generic = {"small", "medium", "large", "single", "double", "regular", "slice", "can", "bottle", "single scoop", "double scoop","big"}
        sizes.update(generic)
        return any(s in text_lower for s in sizes)

    def _looks_like_quantity(self, text_lower: str) -> bool:
        if re.search(r"\b([1-9])\b", text_lower):
            return True
        return any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in NUM_WORDS)

    def _looks_like_address(self, text: str) -> bool:
        return bool(re.search(r"\d+\s+\w+", text)) or any(k in text.lower() for k in {"street", "road"}) or re.search(r"\bng\d", text.lower())

    def _extract_size(self, text_lower: str, item: str = None):
        if item and item in MENU_ITEMS:
            for s in MENU_ITEMS[item]["sizes"].keys():
                if re.search(r"\b" + re.escape(s) + r"\b", text_lower):
                    return s

        candidates = sorted({s for v in MENU_ITEMS.values() for s in v["sizes"]}, key=lambda x: -len(x))
        for s in candidates:
            if re.search(r"\b" + re.escape(s) + r"\b", text_lower):
                return s

        m = re.search(r"\b(\d+-piece)\b", text_lower)
        if m:
            return m.group(1)

        return None

    def _extract_quantity(self, text_lower: str):
        qty_matches = re.findall(r"\b(\d+)\b", text_lower)
        for num in qty_matches:
            if f"{num}-piece" not in text_lower:
                return int(num)

        for w, n in NUM_WORDS.items():
            if re.search(r"\b" + re.escape(w) + r"\b", text_lower):
                return n

        return 1

    def _extract_address(self, text: str):
        m = re.search(r"(?:deliver(?: to)?|delivery to|send to|address is|at)\s+(.+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip(" .")
        return None

    def _allowed_sizes_for(self, item: str):
        if item and item in MENU_ITEMS:
            return list(MENU_ITEMS[item]["sizes"].keys())

        sizes = set()
        for v in MENU_ITEMS.values():
            sizes.update(v["sizes"].keys())
        return sorted(sizes)


    # ORDER SUMMARIES

    def _order_totals_text(self):
        lines = []
        for it in self.order["items"]:
            item = it["item"]
            size = it["size"]
            qty = it["quantity"]
            price_each = MENU_ITEMS[item]["sizes"].get(size, 0.0)
            total = price_each * qty
            lines.append(f"- {qty} × {size} {item} (£{total:.2f})")
        return lines

    def _summarise_order(self):
        if not self.order["items"]:
            return "You have no items in your order."

        lines = self._order_totals_text()
        summary = "Here’s your current order:\n" + "\n".join(lines)

        summary += f"\n\nMethod: {self.order.get('method') or '(not selected)'}"

        if self.order.get("address"):
            summary += f"\nAddress: {self.order['address']}"

        grand = sum(
            MENU_ITEMS[it["item"]]["sizes"].get(it["size"], 0.0) * it["quantity"]
            for it in self.order["items"]
        )

        summary += f"\n\nTotal: £{grand:.2f}"
        return summary

    def last_order_summary_snapshot(self):
        if not self.order["items"]:
            return self.last_order

        lines = self._order_totals_text()
        method = self.order.get("method") or "(method not specified)"
        addr = self.order.get("address") or ""
        addr_text = f"\nAddress: {addr}" if addr else ""

        grand = sum(
            MENU_ITEMS[it["item"]]["sizes"].get(it["size"], 0.0) * it["quantity"]
            for it in self.order["items"]
        )

        confirmation = (
            "Order confirmed!\n\n"
            + "\n".join(lines)
            + f"\n\nMethod: {method}{addr_text}\n\n"
            + f"Total: £{grand:.2f}\n\nThank you! 🍔🍕"
        )

        return confirmation

    def get_last_order_summary(self):
        return self.last_order if self.last_order else self._summarise_order()

    def get_menu_text(self):
        lines = []
        for item, data in MENU_ITEMS.items():
            size_prices = ", ".join(f"{size} £{price:.2f}" for size, price in data["sizes"].items())
            lines.append(f"- {item.title()}: {size_prices}")
        return "Here’s our full menu:\n\n" + "\n".join(lines)

    def get_state_hint(self):
        hints = {
            "idle": "You can ask for the menu or start an order anytime.",
            "awaiting_item": "Tell me the item you'd like.",
            "awaiting_size": "Tell me the size you want.",
            "awaiting_quantity": "Tell me how many you'd like.",
            "awaiting_additional": "You can add another item or say delivery/pickup.",
            "awaiting_method": "Say delivery or pickup.",
            "awaiting_address": "Please provide the delivery address.",
            "awaiting_confirmation": "Say yes to confirm or no to modify your order.",
        }
        return hints.get(self.state, "")
