import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing. Create a .env file with: GROQ_API_KEY=your_key")

PAPER_DIR = "./paper"
INDEX_DIR = "./faiss_index"
MAX_HISTORY_TURNS = 3   # number of recent Q&A turns included in the prompt
TOP_K = 6

# 1. EMBEDDINGS + VECTOR STORE (saved to disk so it is not rebuilt every run)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True},
)

if Path(INDEX_DIR).exists():
    print("Loading existing index...")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    print("Loading documents...")
    loader = DirectoryLoader(
        PAPER_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader,
        show_progress=True, use_multithreading=True, silent_errors=True,
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No documents could be loaded from {PAPER_DIR}")
    print(f"Loaded {len(docs)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks. Creating embeddings...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_DIR)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

# 2. PROMPT
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a friendly research assistant who helps the user understand the scientific papers they uploaded.

How to answer:
- Always answer in English, in a natural, conversational tone that is easy to follow. Avoid sounding robotic.
- Lead with the direct answer, then add explanation only if useful. Never open with phrases like "Based on the context...".
- Synthesize ideas from multiple passages in your own words instead of copying them. Use bullet points when listing things.
- When citing specific information, add a short source tag like (file_name, page X).
- Use only the information in the Documents section. If it is not enough to answer, say plainly that you could not find this in the papers and suggest another angle to ask about. Never make things up.
- Use the Chat history to understand follow-up questions, but do not repeat what was already said.
- If appropriate, end with one short suggested follow-up question.

Chat history:
{history}

Documents:
{context}

Question: {question}

Answer:"""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Using the chat history, rewrite the new question as a complete, standalone question
(replace pronouns such as "it", "that paper", "this method" with the specific subject).
Return only the rewritten question, with no explanation.

History:
{history}

New question: {question}

Rewritten question:"""
)

llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.3)

# 3. CHAT HISTORY
history = []  # list of (question, answer)


def history_text():
    recent = history[-MAX_HISTORY_TURNS:]
    if not recent:
        return "(none yet)"
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)


def format_docs(docs):
    parts = []
    for d in docs:
        name = Path(d.metadata.get("source", "?")).name
        page = d.metadata.get("page", 0) + 1
        parts.append(f"[{name}, page {page}]\n{d.page_content}")
    return "\n\n".join(parts)


def ask(question):
    # Rewrite follow-up questions as standalone ones for better retrieval
    standalone = question
    if history:
        msg = REWRITE_PROMPT.format_messages(history=history_text(), question=question)
        standalone = llm.invoke(msg).content.strip() or question

    docs = retriever.invoke(standalone)
    msg = ANSWER_PROMPT.format_messages(
        history=history_text(), context=format_docs(docs), question=standalone
    )

    answer = ""
    for chunk in llm.stream(msg):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            answer += chunk.content
    print()

    sources = sorted({(Path(d.metadata.get("source", "?")).name, d.metadata.get("page", 0) + 1) for d in docs})
    print("\nSources: " + "; ".join(f"{n} (p.{p})" for n, p in sources))

    history.append((question, answer))


# 4. CHAT LOOP
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