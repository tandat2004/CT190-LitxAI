from rag_core import llm, prepare_answer

history = []  


def ask(question):
    messages, sources = prepare_answer(question, history)

    answer = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            answer += chunk.content
    print()

    print("\nSources: " + "; ".join(f"{n} (p.{p})" for n, p in sources))
    history.append((question, answer))


if __name__ == "__main__":
    print("\nHi! Ask me anything about your papers. Type 'clear' to reset history, 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "clear":
            history.clear()
            print("Chat history cleared.\n")
            continue
        print("Assistant: ", end="")
        ask(q)
        print()
