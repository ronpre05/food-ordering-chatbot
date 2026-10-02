import streamlit as st

from chatbot import Chatbot

# Web front end for the chatbot.
# Run locally with:  streamlit run streamlit_app.py

GREETING = (
    "Hi! I can take a food order, answer questions and chat. "
    "Ask for the menu or tell me what you'd like."
)

EXAMPLES = [
    "What's on the menu?",
    "I want two large pepperoni pizzas and a cola",
    "Who invented pizza?",
    "My name is Sam",
]


def as_markdown(text: str) -> str:
    """Keep the bot's single line breaks when rendered as markdown."""
    return text.replace("\n", "  \n")


def reset_chat():
    st.session_state.bot = Chatbot()
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]


def send(text: str):
    st.session_state.messages.append({"role": "user", "content": text})
    reply = st.session_state.bot.respond(text)
    st.session_state.messages.append({"role": "assistant", "content": reply})


st.set_page_config(page_title="Food Ordering Chatbot", page_icon="🍕")

# One chatbot per browser session, so each visitor has their own order and memory
if "bot" not in st.session_state:
    reset_chat()

st.title("Food Ordering Chatbot")
st.caption(
    "Built with TF-IDF, logistic regression and a finite state machine. No large language model."
)

# Handle new input before drawing the sidebar, so the order panel is up to date
if prompt := st.chat_input("Type a message"):
    send(prompt)

with st.sidebar:
    st.header("Try saying")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            send(example)

    st.divider()
    order = st.session_state.bot.order_system
    st.header("Current order")
    st.write(f"State: `{order.state}`")
    if order.state == "idle":
        st.write("No active order")
    else:
        for line in order.order["items"]:
            st.write(f"{line['quantity']} × {line['size']} {line['item']}")

    st.divider()
    if st.button("Start over", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown("[Source code on GitHub](https://github.com/ronpre05/food-ordering-chatbot)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(as_markdown(message["content"]))
