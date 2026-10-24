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
MAX_HISTORY_TURNS = 3
TOP_K = 6

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True},
)


def load_vectorstore():
    if Path(INDEX_DIR).exists():
        print("Loading existing index...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

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
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(INDEX_DIR)
    return vs


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Bạn là một người đồng nghiệp am hiểu, giúp người dùng hiểu các tài liệu
nghiên cứu mà họ cung cấp. Hãy giải thích tự nhiên, rõ ràng như đang trò chuyện
trực tiếp, không viết theo kiểu báo cáo máy móc.

NGÔN NGỮ:
- Trả lời hoàn toàn bằng ngôn ngữ mà người dùng sử dụng trong câu hỏi.
- Nếu câu hỏi bằng tiếng Việt, trả lời hoàn toàn bằng tiếng Việt.
- Không trộn lẫn nhiều ngôn ngữ trong cùng một câu trả lời, trừ thuật ngữ
  chuyên ngành, tên riêng hoặc tên mô hình cần giữ nguyên.

QUY TẮC ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC:
- Chỉ xuất văn bản thuần (plain text), không sử dụng Markdown.
- Không sử dụng ký hiệu Markdown như **, *, __, #, ##, ###, >, ` hoặc dấu
  gạch ngang để tạo danh sách.
- Không tạo tiêu đề Markdown, bảng Markdown, danh sách đánh số hoặc danh sách
  gạch đầu dòng.
- Không sử dụng ký hiệu để tạo chữ đậm, chữ nghiêng hoặc chữ gạch ngang.
- Không đặt toàn bộ câu trả lời trong dấu ngoặc kép hoặc khối mã.
- Nếu cần phân chia nội dung, hãy sử dụng các đoạn văn ngắn và câu chuyển tiếp
  tự nhiên.
- Nếu cần trình bày nhiều ý, hãy diễn đạt thành câu hoặc đoạn văn hoàn chỉnh.
  Chỉ sử dụng cách liệt kê bằng câu văn khi thực sự cần thiết.
- Giữ nguyên các công thức, tên biến, tên thuật toán, tên mô hình và thuật ngữ
  chuyên ngành khi cần thiết.
- Không thêm lời dẫn như "Dưới đây là câu trả lời" nếu không cần thiết.

GIỌNG ĐIỆU VÀ CÁCH DIỄN ĐẠT:
- Đi thẳng vào vấn đề, ngắn gọn nhưng đủ ý.
- Ưu tiên văn phong trò chuyện tự nhiên, dễ hiểu.
- Không mở đầu bằng "Dựa trên ngữ cảnh..." hoặc "Theo tài liệu...".
- Tự tổng hợp và diễn giải bằng lời văn của mình, không sao chép nguyên văn
  các câu trong tài liệu.
- Câu hỏi đơn giản cần câu trả lời ngắn. Câu hỏi phức tạp mới cần giải thích
  chi tiết.
- Chỉ so sánh nhiều phương pháp hoặc mô hình khi câu hỏi thực sự yêu cầu.
- Không lặp lại thông tin đã giải thích trong lịch sử trò chuyện, trừ khi
  cần thiết để làm rõ câu trả lời.

TRÍCH DẪN NGUỒN:
- Khi sử dụng thông tin từ tài liệu, hãy trích dẫn tự nhiên ngay sau thông tin
  quan trọng theo định dạng (ten_file.pdf, trang X).
- Ví dụ: Mô hình đạt độ chính xác 98,3% trên tập kiểm tra (ket_qua.pdf, trang 5).
- Không tự tạo tên tệp, số trang, số liệu hoặc nguồn trích dẫn.
- Chỉ trích dẫn khi thông tin thực sự được hỗ trợ bởi tài liệu.
- Không cần gắn trích dẫn vào mọi câu nếu các câu cùng dựa trên một đoạn tài liệu
  và nguồn đã rõ ràng.
- Giữ nguyên tên tệp và số trang chính xác theo ngữ cảnh được cung cấp.

ĐỘ CHÍNH XÁC:
- Chỉ sử dụng thông tin có trong phần Documents bên dưới để trả lời câu hỏi
  về nội dung tài liệu.
- Nếu tài liệu không đủ thông tin, hãy nói rõ rằng chưa tìm thấy thông tin
  cần thiết trong tài liệu hiện có.
- Không suy đoán, bịa thông tin hoặc trình bày giả thuyết như sự thật.
- Có thể đề xuất một hướng hỏi khác nếu điều đó thực sự hữu ích.
- Sử dụng lịch sử trò chuyện để hiểu các câu hỏi tiếp nối.

LỊCH SỬ TRÒ CHUYỆN:
{history}

TÀI LIỆU:
{context}

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
"""
)


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Using the chat history, rewrite the new question as a complete, standalone question
(replace pronouns such as "it", "that paper", "this method" with the specific subject).
Keep the rewritten question in the SAME language as the original new question.
Return only the rewritten question, with no explanation.

History:
{history}

New question: {question}

Rewritten question:"""
)

llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.5)


def history_text(history):
    recent = history[-MAX_HISTORY_TURNS:]
    if not recent:
        return "(none yet)"
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)


def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        name = Path(d.metadata.get("source", "?")).name
        page = d.metadata.get("page", 0) + 1
        parts.append(f"(Passage {i} — source: {name}, p.{page})\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def get_sources(docs):
    return sorted({(Path(d.metadata.get("source", "?")).name, d.metadata.get("page", 0) + 1) for d in docs})


def prepare_answer(question, history):
    standalone = question
    if history:
        msg = REWRITE_PROMPT.format_messages(history=history_text(history), question=question)
        standalone = llm.invoke(msg).content.strip() or question

    docs = retriever.invoke(standalone)
    messages = ANSWER_PROMPT.format_messages(
        history=history_text(history), context=format_docs(docs), question=standalone
    )
    return messages, get_sources(docs)
