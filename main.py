from chatbot import Chatbot


def main():
    """
    Command-line entry point
    """

    print("Welcome to the food ordering chatbot!")
    print("Type 'exit' to quit.\n")

    bot = Chatbot()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        response = bot.respond(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
