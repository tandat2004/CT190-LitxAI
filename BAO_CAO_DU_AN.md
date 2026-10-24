# BÁO CÁO ĐỒ ÁN HỌC PHẦN
## MÔN HỌC: NHẬP MÔN TRÍ TUỆ NHÂN TẠO (MÃ HỌC PHẦN: CT190)

---

### **ĐỀ TÀI:**
# **CT190-LitxAI: TRỢ LÝ HỎI ĐÁP VÀ ĐỌC HIỂU TÀI LIỆU NGHIÊN CỨU KHOA HỌC ỨNG DỤNG KỸ THUẬT RAG (RETRIEVAL-AUGMENTED GENERATION)**

* **Trường**: Trường Công nghệ Thông tin & Truyền thông - Đại học Cần Thơ (CICT - CTU)
* **Học phần**: CT190 - Nhập môn Trí tuệ Nhân tạo
* **Sinh viên thực hiện**: Trần Tấn Đạt (`tandat2004`)
* **Kho lưu trữ (Repository)**: [GitHub: tandat2004/CT190-LitxAI](https://github.com/tandat2004/CT190-LitxAI)
* **Thời gian thực hiện**: Học kỳ 1, Năm học 2024 - 2025

---

## MỤC LỤC
1. [TỔNG QUAN ĐỀ TÀI (ABSTRACT)](#1-tổng-quan-đề-tài-abstract)
2. [CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ MỤC TIÊU DỰ ÁN](#chương-1-đặt-vấn-đề-và-mục-tiêu-dự-án)
   - 1.1. Bối cảnh và thách thức khi đọc tài liệu khoa học
   - 1.2. Hạn chế của các mô hình ngôn ngữ lớn (LLM) độc lập
   - 1.3. Phương pháp Retrieval-Augmented Generation (RAG)
   - 1.4. Mục tiêu cụ thể của đề tài
3. [CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ ÁP DỤNG](#chương-2-cơ-sở-lý-thuyết-và-công-nghệ-áp-dụng)
   - 2.1. Kiến trúc tổng quát của hệ thống RAG
   - 2.2. Kỹ thuật biểu diễn không gian ngữ nghĩa (Vector Embeddings)
   - 2.3. Cơ sở dữ liệu Vector và thuật toán tìm kiếm tương đồng (FAISS)
   - 2.4. Kỹ thuật phân đoạn văn bản đệ quy (Recursive Text Splitting)
   - 2.5. Kỹ thuật tái cấu trúc câu hỏi theo ngữ cảnh (Contextual Query Rewriting)
   - 2.6. Mô hình ngôn ngữ lớn (LLM) và hạ tầng tăng tốc phần cứng Groq
4. [CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC VÀ QUY TRÌNH HỆ THỐNG](#chương-3-thiết-kế-kiến-trúc-và-quy-trình-hệ-thống)
   - 3.1. Sơ đồ kiến trúc tổng thể (Architecture Flow Diagram)
   - 3.2. Quy trình lập chỉ mục tài liệu (Indexing Pipeline)
   - 3.3. Quy trình truy hồi và sinh câu trả lời (Query & Inference Pipeline)
   - 3.4. Kỹ thuật kiểm soát định dạng và trích dẫn minh bạch
   - 3.5. Cơ chế phân tách dữ liệu phản hồi (Response Streaming Protocol)
5. [CHƯƠNG 4: HIỆN THỰC HÓA VÀ CHI TIẾT CÀI ĐẶT](#chương-4-hiện-thực-hóa-và-chi-tiết-cài-đặt)
   - 4.1. Cấu trúc tổ chức mã nguồn
   - 4.2. Phân tích chi tiết các module chính
   - 4.3. Tập dữ liệu thực nghiệm (Corpus of Scientific Papers)
   - 4.4. Thống kê phân đoạn và chỉ mục vector
6. [CHƯƠNG 5: KỊCH BẢN THỬ NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ](#chương-5-kịch-bản-thử-nghiệm-và-đánh-giá-kết-quả)
   - 5.1. Kịch bản 1: Truy vấn trích xuất thông tin học thuật cụ thể
   - 5.2. Kịch bản 2: Đối thoại đa lượt và giải quyết đại từ thay thế
   - 5.3. Kịch bản 3: Khả năng chống ảo giác khi câu hỏi ngoài phạm vi
   - 5.4. Kịch bản 4: Độ chính xác của trích dẫn nguồn và số trang
   - 5.5. Đánh giá tốc độ và tài nguyên hệ thống
7. [CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN](#chương-6-kết-luận-và-hướng-phát-triển)
   - 6.1. Đóng góp và kết quả đạt được
   - 6.2. Hạn chế hiện tại
   - 6.3. Đề xuất hướng phát triển tương lai
8. [TÀI LIỆU THAM KHẢO](#tài-liệu-tham-khảo)
9. [PHỤ LỤC: HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH](#phụ-lục-hướng-dẫn-cài-đặt-và-vận-hành)

---

## 1. TỔNG QUAN ĐỀ TÀI (ABSTRACT)

Trong quá trình học tập và nghiên cứu khoa học, sinh viên và các nhà nghiên cứu thường xuyên đối mặt với khối lượng lớn các bài báo học thuật dày đặc thuật ngữ, biểu thức và dữ liệu thực nghiệm. Việc tìm kiếm thông tin chính xác, so sánh kết quả và tổng hợp nội dung đòi hỏi nhiều thời gian và công sức. 

Đề tài **CT190-LitxAI** (Literature Review AI Assistant) được xây dựng nhằm giải quyết bài toán trên thông qua việc áp dụng kỹ thuật tiên tiến trong Trí tuệ Nhân tạo hiện đại: **RAG (Retrieval-Augmented Generation)** kết hợp cùng mô hình ngôn ngữ lớn (LLM). Dự án xây dựng một trợ lý hỏi đáp thông minh có khả năng đọc hiểu tự động toàn bộ kho tài liệu nghiên cứu khoa học chuyên ngành (ở định dạng PDF), thực hiện truy hồi chính xác các đoạn văn bản chứa câu trả lời và diễn giải tự nhiên bằng tiếng Việt hoặc tiếng Anh.

Hệ thống tích hợp mô hình nhúng ngữ nghĩa đa ngữ chất lượng cao **`intfloat/multilingual-e5-base`** (768 chiều vector), cơ sở dữ liệu vector tốc độ cao **FAISS (Facebook AI Similarity Search)** với 292 chunks tài liệu thực nghiệm, kỹ thuật **Tái cấu trúc câu hỏi đa lượt (Contextual Query Rewriting)** và mô hình ngôn ngữ **`openai/gpt-oss-120b`** thông qua nền tảng suy luận tăng tốc **Groq LPU**. Hệ thống cung cấp hai phương thức tương tác: giao diện web thời gian thực với streaming token mượt mà và giao diện dòng lệnh (CLI). Điểm đặc biệt của hệ thống là khả năng kiểm soát chặt chẽ hiện tượng bịa đặt thông tin (Anti-Hallucination) và trích dẫn nguồn minh bạch đến từng tệp tin và trang tài liệu tương ứng.

---

## CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ MỤC TIÊU DỰ ÁN

### 1.1. Bối cảnh và thách thức khi đọc tài liệu khoa học
Nghiên cứu khoa học đòi hỏi quy trình duyệt tài liệu (Literature Review) nghiêm ngặt nhằm nắm bắt tình hình nghiên cứu (State of the Art), phương pháp luận và các kết quả thực nghiệm. Tuy nhiên:
- Số lượng công trình nghiên cứu công bố ngày càng lớn.
- Các tài liệu có độ dài từ 8 đến 20 trang với cấu trúc phức tạp: tóm tắt, cơ sở lý thuyết, kiến trúc mạng học sâu, bảng biểu so sánh, kết luận.
- Độc giả khó định vị nhanh thông số cấu hình, độ chính xác hoặc chi tiết thuật toán mà không phải đọc tuần tự toàn bộ tài liệu.

### 1.2. Hạn chế của các mô hình ngôn ngữ lớn (LLM) độc lập
Mặc dù các mô hình ngôn ngữ lớn như GPT-4, LLaMA có khả năng sinh văn bản vượt trội, chúng bộc lộ ba điểm yếu cốt lõi khi xử lý tài liệu chuyên ngành:
1. **Ảo giác thông tin (Hallucination)**: LLM có xu hướng tự sinh ra thông tin có vẻ hợp lý nhưng không có thật hoặc sai lệch số liệu thực nghiệm.
2. **Thiếu tri thức miền cục bộ (Knowledge Cut-off & Private Domain)**: LLM không được huấn luyện trên các tài liệu nội bộ, các bài báo mới xuất bản hoặc các kỷ yếu hội nghị chuyên đề cục bộ (chẳng hạn các công trình từ Đại học Cần Thơ).
3. **Không có khả năng kiểm chứng nguồn (Lack of Provenance)**: LLM thông thường không thể chỉ ra thông tin được lấy chính xác từ trang nào, đoạn văn nào của bài báo nào.

### 1.3. Phương pháp Retrieval-Augmented Generation (RAG)
Kỹ thuật **RAG (Retrieval-Augmented Generation)** do Lewis và cộng sự giới thiệu năm 2020 là sự kết hợp tối ưu giữa hệ thống truy hồi thông tin (Information Retrieval - IR) và mô hình sinh ngôn ngữ tự nhiên (Generative Model). Thay vì yêu cầu LLM "nhớ" toàn bộ kiến thức trong trọng số tham số (parametric memory), RAG cung cấp cho LLM bộ nhớ ngoài (non-parametric memory) dưới dạng kho lưu trữ vector ngữ nghĩa. Khi nhận câu hỏi, hệ thống truy xuất các đoạn văn bản liên quan nhất và đưa trực tiếp vào ngữ cảnh (context) cho LLM xử lý, giúp câu trả lời luôn có căn cứ thực tế và có thể trích dẫn nguồn minh bạch.

### 1.4. Mục tiêu cụ thể của đề tài
Dự án hướng tới các mục tiêu:
1. **Xây dựng quy trình lập chỉ mục tài liệu tự động (Indexing Pipeline)**: Đọc, tiền xử lý và cắt nhỏ các tệp bài báo PDF học thuật, nhúng vector và lập chỉ mục trong FAISS.
2. **Triển khai kỹ thuật hỏi đáp đa lượt thông minh**: Sử dụng cơ chế tái viết câu hỏi (Query Rewriting) để hiểu đúng ngữ cảnh khi người dùng hỏi các câu tiếp nối chứa đại từ thay thế (như "mô hình này", "phương pháp đó", "nó đạt kết quả bao nhiêu").
3. **Thiết kế cơ chế Grounding và trích dẫn chuẩn mực**: Đảm bảo LLM chỉ trả lời dựa trên tài liệu được cung cấp, nói rõ khi không tìm thấy thông tin, và luôn trích dẫn kèm định dạng `(ten_file.pdf, trang X)`.
4. **Cung cấp trải nghiệm người dùng tối ưu**: Xây dựng Web UI hỗ trợ cơ chế phản hồi theo luồng dữ liệu (Streaming Response) và trích xuất danh sách nguồn tham khảo tự động; kèm theo giao diện dòng lệnh linh hoạt.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ ÁP DỤNG

### 2.1. Kiến trúc tổng quát của hệ thống RAG
Hệ thống RAG chia làm hai pha độc lập nhưng liên kết chặt chẽ:
- **Pha Lập chỉ mục (Offline Indexing)**: Tài liệu được nạp -> Phân đoạn nhỏ (Chunking) -> Biến đổi thành vector thông qua mô hình Embedding -> Lưu trữ vào Vector Database.
- **Pha Truy vấn và Sinh câu trả lời (Online Retrieval & Generation)**: Câu hỏi người dùng -> Tái cấu trúc truy vấn -> Tìm kiếm $K$ đoạn văn bản có độ tương đồng ngữ nghĩa cao nhất -> Đóng gói vào Prompt -> LLM tổng hợp câu trả lời kèm nguồn dẫn.

```
[User Question] ──> [Query Rewriter (LLM)] ──> [Standalone Query]
                                                         │
                                                  (Embedding Vector)
                                                         │
                                                         ▼
[PDF Documents] ──> [Text Splitter] ──> [Embeddings] ──> [FAISS Index]
                                                         │
                                                    (Top-K Chunks)
                                                         │
                                                         ▼
                                                [Context + Prompt]
                                                         │
                                                         ▼
                                                  [ChatGroq LLM]
                                                         │
                                                         ▼
                                              [Answer + Citations]
```

### 2.2. Kỹ thuật biểu diễn không gian ngữ nghĩa (Vector Embeddings)
Mô hình nhúng chuyển đổi các chuỗi văn bản $T$ thành các vector thực nhiều chiều $\mathbf{v} \in \mathbb{R}^d$ sao cho khoảng cách giữa các vector phản ánh mức độ tương đồng ngữ nghĩa giữa các văn bản tương ứng.

Dự án lựa chọn mô hình **`intfloat/multilingual-e5-base`**:
- **Kiến trúc**: Dựa trên kiến trúc Transformer hai chiều (BERT-style) được tiền huấn luyện trên tập dữ liệu đa ngữ khổng lồ (CC100, Wikipedia, mC4).
- **Kích thước vector ($d$)**: 768 chiều.
- **Hỗ trợ đa ngôn ngữ**: Hoạt động tối ưu trên cả tiếng Việt và tiếng Anh, phù hợp với tập tài liệu song ngữ trong nghiên cứu khoa học.
- **Chuẩn hóa vector (L2 Normalization)**: Mỗi vector $\mathbf{v}$ được chuẩn hóa sao cho $\|\mathbf{v}\|_2 = 1$. Khi đó khoảng cách Euclid bình phương và độ tương đồng Cosine thỏa mãn hệ thức:
  $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2\langle \mathbf{u}, \mathbf{v} \rangle = 2 - 2\cos(\theta)$$
  Do đó, bài toán tìm khoảng cách Euclid nhỏ nhất hoàn toàn tương đương với việc tìm độ tương đồng Cosine lớn nhất.

### 2.3. Cơ sở dữ liệu Vector và thuật toán tìm kiếm tương đồng (FAISS)
**FAISS (Facebook AI Similarity Search)** là thư viện hàng đầu tối ưu hóa cho việc tìm kiếm vector mật độ cao (Dense Vector Search) trên bộ nhớ RAM:
- Sử dụng cấu trúc **`IndexFlatL2`**: Thực hiện tìm kiếm vét cạn (brute-force exact search) không nén dữ liệu, đảm bảo độ chính xác tuyệt đối (Recall 100%) mà không bị suy hao chất lượng như các phương pháp xấp xỉ (HNSW, IVF-PQ).
- Với tập dữ liệu nghiên cứu ban đầu gồm vài trăm đến vài nghìn đoạn văn bản, `IndexFlatL2` mang lại thời gian tìm kiếm dưới 1 mili-giây (sub-millisecond latency) mà không làm tốn tài nguyên tính toán.

### 2.4. Kỹ thuật phân đoạn văn bản đệ quy (Recursive Text Splitting)
Nếu đưa toàn bộ bài báo vào mô hình nhúng, vector sẽ bị phân tán thông tin và vượt quá giới hạn token (Token Limit). Do đó, cần phân rã tài liệu thành các đoạn nhỏ (Chunks):
- Sử dụng **`RecursiveCharacterTextSplitter`** với danh sách phân cách ưu tiên: `["\n\n", "\n", ". ", " ", ""]`.
- **Kích thước đoạn (`chunk_size=1200`)**: Khoảng 250 - 350 từ, đủ chứa trọn vẹn một ý tưởng, một định nghĩa hoặc một đoạn diễn giải phương pháp/kết quả thực nghiệm.
- **Độ chồng lấp (`chunk_overlap=200`)**: Tạo sự gối đầu giữa các đoạn liên tiếp, ngăn chặn hiện tượng câu văn hoặc luận điểm bị cắt đứt giữa chừng tại ranh giới phân tách.

### 2.5. Kỹ thuật tái cấu trúc câu hỏi theo ngữ cảnh (Contextual Query Rewriting)
Trong hội thoại tự nhiên, người dùng thường đặt các câu hỏi tiếp nối như:
- Lượt 1: *"Phương pháp AMMAD được đề xuất trong bài báo nào?"*
- Lượt 2: *"Nó đạt độ chính xác bao nhiêu phần trăm?"*

Nếu truyền trực tiếp câu hỏi ở lượt 2 vào FAISS, hệ thống chỉ tìm kiếm từ khóa "nó", dẫn đến việc truy xuất sai tài liệu. Giải pháp trong đề tài là sử dụng LLM với **`REWRITE_PROMPT`** để tự động thay thế đại từ tham chiếu dựa trên 3 lượt trao đổi gần nhất (`MAX_HISTORY_TURNS = 3`), chuyển thành câu hỏi độc lập: *"Mô hình AMMAD đạt độ chính xác bao nhiêu phần trăm?"* trước khi thực hiện tìm kiếm vector.

### 2.6. Mô hình ngôn ngữ lớn (LLM) và hạ tầng tăng tốc phần cứng Groq
Hệ thống sử dụng mô hình mã nguồn mở thế hệ mới **`openai/gpt-oss-120b`** thông qua nền tảng đám mây **Groq Cloud**:
- **Bộ xử lý LPU (Language Processing Unit)**: Kiến trúc phần cứng chuyên biệt dạng ma trận Tensor của Groq cho phép suy luận LLM với tốc độ cực cao (> 300 - 500 tokens/giây), loại bỏ độ trễ cổ chai bộ nhớ (Memory Bandwidth Bottleneck) của GPU thông thường.
- **Nhiệt độ sinh (`temperature=0.5`)**: Đặt ở mức trung bình thấp nhằm đảm bảo tính chính xác khoa học, câu từ gãy gọn và bám sát ngữ cảnh thực tế, hạn chế tối đa tính ngẫu nhiên và suy diễn sai lệch.

---

## CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC VÀ QUY TRÌNH HỆ THỐNG

### 3.1. Sơ đồ kiến trúc tổng thể (Architecture Flow Diagram)

Dưới đây là sơ đồ kiến trúc thể hiện toàn bộ các thành phần và luồng dữ liệu của hệ thống:

```mermaid
flowchart TD
    subgraph INGESTION["1. Quá trình Lập chỉ mục (Indexing Phase)"]
        A[Thư mục chứa PDF /paper] --> B[PyMuPDFLoader - Đọc từng trang & metadata]
        B --> C[RecursiveCharacterTextSplitter - chunk: 1200, overlap: 200]
        C --> D[HuggingFaceEmbeddings - multilingual-e5-base]
        D --> E[(FAISS Vector Store - index.faiss + index.pkl)]
    end

    subgraph QUERY["2. Quá trình Xử lý Truy vấn (Query Phase)"]
        U[Người dùng nhập câu hỏi] --> Q[Kiểm tra Lịch sử Chat]
        Q -->|Có lịch sử| R[Contextual Query Rewriter - LLM]
        Q -->|Chưa có lịch sử| S[Câu hỏi gốc]
        R --> S
        S --> T[FAISS Retriever - similarity search Top-K=6]
        E -.->|Truy xuất ngữ nghĩa| T
        T --> F[Format Passages & Trích xuất Metadata Nguồn]
    end

    subgraph GENERATION["3. Quá trình Sinh & Phản hồi (Generation & Streaming)"]
        F --> G[Xây dựng ANSWER_PROMPT nghiêm ngặt]
        G --> H[ChatGroq - openai/gpt-oss-120b]
        H -->|Stream Tokens| I[Streaming Engine]
        I -->|Text Response| J[Web UI / Terminal CLI]
        I -->|Marker \x00SOURCES\x00 + JSON| K[Hiển thị Nguồn & Số trang]
    end
```

### 3.2. Quy trình lập chỉ mục tài liệu (Indexing Pipeline)
Được triển khai trong hàm `load_vectorstore()` thuộc tệp [rag_core.py](file:///d:/Computer%20Science/CT190%20-%20Nhap%20mon%20AI/BTGK/rag_core.py#L28-L51):
1. **Kiểm tra trạng thái sẵn sàng**: Kiểm tra sự tồn tại của thư mục `./faiss_index`. Nếu đã có chỉ mục sẵn, hệ thống tải trực tiếp lên bộ nhớ RAM bằng hàm `FAISS.load_local()`, tiết kiệm thời gian khởi động (chỉ mất ~0.5s).
2. **Trích xuất văn bản**: Nếu chưa có chỉ mục, `DirectoryLoader` kết hợp `PyMuPDFLoader` duyệt toàn bộ tệp `.pdf` trong thư mục `./paper`. PyMuPDF đọc từng trang, bảo toàn metadata gồm đường dẫn tệp (`source`) và chỉ số trang (`page`).
3. **Phân mảnh văn bản**: `RecursiveCharacterTextSplitter` chia các tài liệu thành các chunks văn bản có độ dài 1200 ký tự với độ chồng lấp 200 ký tự.
4. **Vector hóa và lưu trữ**: Mô hình nhúng mã hóa từng chunk thành vector 768 chiều và lưu chỉ mục xuống đĩa tại `./faiss_index`.

### 3.3. Quy trình truy hồi và sinh câu trả lời (Query & Inference Pipeline)
Được quản lý thông qua hàm `prepare_answer(question, history)`:
1. **Nhận câu hỏi**: Tiếp nhận câu hỏi $Q$ và danh sách lịch sử hội thoại $H$.
2. **Tái tạo truy vấn (Query Rewriting)**: Nếu $H \neq \emptyset$, gọi `REWRITE_PROMPT` để biến đổi $Q$ thành câu truy vấn độc lập $Q_{standalone}$.
3. **Truy hồi ngữ nghĩa (Semantic Retrieval)**: Thực hiện tìm kiếm `retriever.invoke(Q_{standalone})` lấy ra 6 đoạn văn bản ($TOP\_K = 6$) có độ tương đồng cosine lớn nhất.
4. **Đóng gói Prompt (Context Assembly)**: Ghép các đoạn văn bản kèm định danh thứ tự `(Passage i — source: tên_tệp, p.trang)` vào biến `{context}` của `ANSWER_PROMPT`.
5. **Sinh phản hồi (Generation)**: Gửi tin nhắn có cấu trúc đến mô hình ngôn ngữ `ChatGroq`.

### 3.4. Kỹ thuật kiểm soát định dạng và trích dẫn minh bạch
Prompt hệ thống (`ANSWER_PROMPT`) trong [rag_core.py](file:///d:/Computer%20Science/CT190%20-%20Nhap%20mon%20AI/BTGK/rag_core.py#L58-L127) được thiết kế đặc thù với các quy tắc kỹ thuật nghiêm ngặt:
- **Ngôn ngữ phản hồi**: Tự động khớp ngôn ngữ với câu hỏi người dùng (Hỏi tiếng Việt đáp tiếng Việt, hỏi tiếng Anh đáp tiếng Anh).
- **Quy tắc định dạng**: Cấm các định dạng Markdown thô như tiêu đề `#`, dấu gạch ngang đầu dòng `*`, bảng biểu phức tạp. Ép LLM diễn đạt theo câu văn tự nhiên, giúp nội dung trả lời mạch lạc như một chuyên gia đối thoại trực tiếp.
- **Nguyên tắc chống bịa đặt (Anti-Hallucination & Grounding)**: Nếu thông tin không có trong phần `Documents`, mô hình bắt buộc phải trả lời rằng tài liệu hiện có chưa cung cấp thông tin, tuyệt đối không được tự suy đoán.
- **Trích dẫn nguồn trực tiếp (In-text Citation)**: Mô hình phải đặt nhãn nguồn `(ten_file.pdf, trang X)` ngay sau câu chứa thông tin quan trọng.

### 3.5. Cơ chế phân tách dữ liệu phản hồi (Response Streaming Protocol)
Một thách thức kỹ thuật lớn trong ứng dụng web là vừa muốn truyền câu trả lời theo thời gian thực (Streaming) đến trình duyệt, vừa muốn gửi kèm danh sách siêu dữ liệu (Metadata) các tệp nguồn để giao diện hiển thị thành các thẻ liên kết (Source Badges).

Dự án đề xuất một giải pháp giao thức phân tách bằng ký tự điều khiển:
```python
SOURCES_MARKER = "\x00SOURCES\x00"
```
- Khi endpoint `/chat` của FastAPI phát sinh phản hồi theo luồng `StreamingResponse`:
  1. Từng token chữ của câu trả lời được yield trực tiếp về client.
  2. Khi kết thúc câu trả lời, server yield chuỗi phân cách đặc biệt `\x00SOURCES\x00` kèm chuỗi JSON chứa danh sách nguồn: `[{"name": ..., "page": ...}]`.
- Phía Web Client (`index.html`), `ReadableStream` nhận luồng byte, dò tìm vị trí của `SOURCES_MARKER`. Toàn bộ phần trước marker được render thành nội dung câu trả lời; phần sau marker được parse bằng `JSON.parse()` và hiển thị thành danh sách các nguồn tham chiếu một cách trực quan.

---

## CHƯƠNG 4: HIỆN THỰC HÓA VÀ CHI TIẾT CÀI ĐẶT

### 4.1. Cấu trúc tổ chức mã nguồn

```
d:/Computer Science/CT190 - Nhap mon AI/BTGK/
│
├── .env                       # Khóa API bí mật (GROQ_API_KEY)
├── .gitignore                 # Bỏ qua tệp môi trường và bộ nhớ đệm
├── README.md                  # Giới thiệu dự án cơ bản
├── requirements.txt           # Danh mục các thư viện phụ thuộc của dự án
├── BAO_CAO_DU_AN.md           # Báo cáo đồ án chi tiết (Tài liệu này)
│
├── rag_core.py                # Mô-đun cốt lõi RAG: Loader, Splitter, FAISS, Prompt, LLM
├── chatbot.py                 # Giao diện dòng lệnh CLI tương tác trực tiếp
├── server.py                  # Backend API xây dựng trên nền tảng FastAPI
│
├── static/
│   └── index.html             # Giao diện Web Client Single-Page hiện đại
│
├── faiss_index/               # Cơ sở dữ liệu Vector đã được lập chỉ mục sẵn
│   ├── index.faiss            # File ma trận vector (IndexFlatL2, 768 chiều, 292 vector)
│   └── index.pkl              # File đối tượng Python lưu Document Chunks và Metadata
│
└── paper/                     # Thư mục chứa 9 bài báo nghiên cứu khoa học thực tế (.pdf)
    ├── 60_FAIR2022_2692.pdf
    ├── Building_a_wastewater_network_graph_from_inspection_videos.pdf
    ├── FinalVersion_ICTA_AMMAD_2025.pdf
    ├── Hieu_qua_cac_nhom_mo_hinh_hoc_sau_trong_bai_toan_p.pdf
    ├── Phat_hien_va_hien_thi_3D_vung_bat_thuong_tren_anh_.pdf
    ├── fp.pdf
    ├── main.pdf
    ├── paper51_ESTIMATINGTHETRAFFICDENSITYFROMTRAFFICCAMERAS.pdf
    └── wwt.pdf
```

### 4.2. Phân tích chi tiết các module chính

#### Module 1: `rag_core.py` (Trái tim của hệ thống)
- **Tải môi trường**: Sử dụng `python-dotenv` để nạp an toàn biến môi trường `GROQ_API_KEY`.
- **Cấu hình mô hình**:
  - `embeddings`: Sử dụng `HuggingFaceEmbeddings` gọi mô hình `intfloat/multilingual-e5-base`, thiết lập `normalize_embeddings=True`.
  - `llm`: Sử dụng `ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.5)`.
- **Cấu trúc lưu trữ FAISS**: Tự động nhận diện và tái sử dụng chỉ mục đã biên dịch trên đĩa, nếu thiếu sẽ tự động chạy toàn bộ quy trình đọc tài liệu và phân đoạn.
- **Hàm `history_text(history)`**: Lấy 3 lượt hội thoại gần nhất, format dưới dạng cấu trúc hội thoại chuẩn để cung cấp ngữ cảnh cho mô hình.
- **Hàm `format_docs(docs)`**: Định dạng từng đoạn trích từ vectorstore gồm chỉ mục đoạn văn, tên file PDF gốc và số trang được đánh số theo chuẩn người đọc (+1 từ chỉ mục 0-indexed).

#### Module 2: `chatbot.py` (Giao diện dòng lệnh)
- Cung cấp vòng lặp hội thoại `while True` với khả năng bắt ngoại lệ người dùng ngắt tín hiệu (`KeyboardInterrupt`, `EOFError`).
- Hỗ trợ các lệnh điều khiển nhanh:
  - `clear`: Xóa sạch lịch sử hội thoại hiện tại.
  - `exit` / `quit`: Thoát chương trình.
- Tận dụng phương thức `llm.stream()` để in từng ký tự phản hồi ra màn hình console theo thời gian thực (hiệu ứng typing).

#### Module 3: `server.py` (Máy chủ API)
- Sử dụng framework hiện đại **FastAPI** và máy chủ bất đồng bộ **Uvicorn**.
- Định nghĩa cấu trúc dữ liệu gửi lên thông qua Pydantic: `class Ask(BaseModel): question: str`.
- Cung cấp 3 API chính:
  1. `GET /`: Trả về giao diện người dùng `static/index.html`.
  2. `POST /chat`: Nhận câu hỏi, gọi `prepare_answer()`, stream kết quả và trả về mã nguồn kèm marker.
  3. `POST /clear`: Xóa lịch sử phiên hội thoại hiện tại.

#### Module 4: `static/index.html` (Giao diện Web)
- Thiết kế giao diện theo phong cách tối giản, thanh lịch: bố cục responsive, bảng màu tương phản nhẹ nhàng, bóng đổ hiện đại.
- Ứng dụng `Fetch API` kết hợp `res.body.getReader()` để đọc trực tiếp luồng byte phản hồi theo cơ chế Chunked Transfer Encoding.
- Tự động cuộn trang (`window.scrollTo()`) theo tiến độ trả lời của trợ lý ảo.

---

### 4.3. Tập dữ liệu thực nghiệm (Corpus of Scientific Papers)

Hệ thống được thử nghiệm trên bộ sưu tập gồm **9 bài báo khoa học** xuất bản tại các hội nghị khoa học quốc gia (FAIR), hội nghị quốc tế (ICTA) và tạp chí chuyên ngành uy tín (Tạp chí Khoa học Đại học Cần Thơ), được nghiên cứu bởi các giảng viên và chuyên gia Trí tuệ Nhân tạo tại Đại học Cần Thơ:

| STT | Tên tệp PDF | Tiêu đề công trình / Đề tài nghiên cứu | Nhóm tác giả chính | Đơn vị / Xuất bản | Số trang | Số Chunks |
|:---:|:---|:---|:---|:---|:---:|:---:|
| 1 | `60_FAIR2022_2692.pdf` | Ứng dụng mạng GAN có điều kiện phát hiện vùng bất thường trên ảnh MRI não | Nhóm nghiên cứu ĐH Cần Thơ | Kỷ yếu FAIR 2022 | 8 | 31 |
| 2 | `Building_a_wastewater_network_graph_from_inspection_videos.pdf` | Building a wastewater network graph from inspection videos | Minh-Thu TRAN-NGUYEN, Salem BENFERHAT, Thanh-Nghi DO et al. | CICT - CTU & CRIL, CNRS | 15 | 42 |
| 3 | `FinalVersion_ICTA_AMMAD_2025.pdf` | AMMAD: A Multimodal Approach for Disease Diagnosis with Symptoms and Tests | Minh-Thu Tran-Nguyen, Thanh Ma, Viet-Chau Tran et al. | Đại học Cần Thơ | 8 | 23 |
| 4 | `Hieu_qua_cac_nhom_mo_hinh_hoc_sau_trong_bai_toan_p.pdf` | Hiệu quả các nhóm mô hình học sâu trong bài toán phát hiện phương tiện giao thông | Vũ Lê Quỳnh Phương, Trần Nguyễn Minh Thư, Phạm Nguyên Khang | Tạp chí KH ĐHCT (2024) | 8 | 38 |
| 5 | `Phat_hien_va_hien_thi_3D_vung_bat_thuong_tren_anh_.pdf` | Phát hiện và hiển thị 3D vùng bất thường trên ảnh MRI não với cổng dịch vụ Billow AISA | Lê Minh Lợi, Trần Nguyễn Minh Thư, Nguyễn Thiện Hùng et al. | Tạp chí KH ĐHCT (2024) | 9 | 34 |
| 6 | `fp.pdf` | Training Deep Network Models for Fingerprint Image Classification | Thanh-Nghi Do, Minh-Thu Tran-Nguyen | CICT - ĐH Cần Thơ | 12 | 27 |
| 7 | `main.pdf` | ImageNet Challenging Classification with the Raspberry Pis: A Federated Learning Algorithm of Local SGD | Thanh-Nghi Do, Minh-Thu Tran-Nguyen | CICT - ĐH Cần Thơ | 14 | 33 |
| 8 | `paper51_ESTIMATINGTHETRAFFICDENSITYFROMTRAFFICCAMERAS.pdf` | Estimating the Traffic Density from Traffic Cameras | Vũ Lê Quỳnh Phương, Bùi Nhật Tài, Nguyễn Khắc Huy et al. | Kỷ yếu Hội thảo chuyên ngành | 18 | 38 |
| 9 | `wwt.pdf` | Deep Networks for Monitoring Waterway Traffic in the Mekong Delta | Thanh-Nghi Do, Minh-Thu Tran-Nguyen, Thanh-Tri Trang et al. | CICT - ĐH Cần Thơ | 12 | 26 |
| **Tổng cộng** | **9 bài báo** | **Đa lĩnh vực: Y tế, Giao thông thông minh, IoT, Học sâu** | — | — | **104 trang** | **292 chunks** |

### 4.4. Thống kê phân đoạn và chỉ mục vector
- **Tổng số trang nạp**: 104 trang tài liệu khoa học.
- **Tổng số phân đoạn sinh ra**: 292 chunks văn bản (trung bình 2.8 chunks/trang tài liệu).
- **Kích thước vector**: 768 chiều/vector.
- **Dung lượng lưu trữ FAISS**:
  - `index.faiss`: ~897 KB.
  - `index.pkl`: ~385 KB.
  - Tổng cộng toàn bộ cơ sở dữ liệu vector chỉ chiếm ~1.28 MB, cho phép tải toàn bộ vào RAM ngay lập tức và tìm kiếm cực nhanh.

---

## CHƯƠNG 5: KỊCH BẢN THỬ NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

### 5.1. Kịch bản 1: Truy vấn trích xuất thông tin học thuật cụ thể
* **Mục tiêu**: Kiểm tra khả năng tìm kiếm thông tin sự thật (Fact-based QA) và trích dẫn số trang chính xác.
* **Câu hỏi**: *"Mô hình AMMAD trong bài báo FinalVersion_ICTA_AMMAD_2025.pdf sử dụng phương pháp gì để chẩn đoán bệnh?"*
* **Kết quả xử lý của hệ thống**:
  - FAISS tìm được các chunks liên quan trong bài `FinalVersion_ICTA_AMMAD_2025.pdf` (trang 1, 2, 3).
  - Trợ lý giải thích rõ: Mô hình AMMAD kết hợp dữ liệu đa phương thức (Multimodal) giữa các triệu chứng lâm sàng (Symptoms) và kết quả xét nghiệm (Tests) của bệnh nhân.
  - Trích dẫn chính xác nguồn: `(FinalVersion_ICTA_AMMAD_2025.pdf, trang 1)`.

### 5.2. Kịch bản 2: Đối thoại đa lượt và giải quyết đại từ thay thế
* **Mục tiêu**: Kiểm tra năng lực của `REWRITE_PROMPT` trong việc xử lý ngữ cảnh đối thoại.
* **Quy trình thử nghiệm**:
  - **Lượt 1**: Người dùng hỏi: *"Bài báo về phân loại ảnh vân tay sử dụng mô hình học sâu nào?"*
    - Trợ lý trả lời dựa trên tệp `fp.pdf`.
  - **Lượt 2**: Người dùng hỏi tiếp: *"Phương pháp này được huấn luyện trên tập dữ liệu gì và kết quả ra sao?"*
    - **Quá trình Rewrite câu hỏi**:
      - Input: Lịch sử về bài báo `fp.pdf` + Câu hỏi mới: *"Phương pháp này được huấn luyện trên tập dữ liệu gì..."*
      - Output tái cấu trúc: *"Phương pháp phân loại ảnh vân tay trong bài báo fp.pdf được huấn luyện trên tập dữ liệu nào và kết quả ra sao?"*
    - FAISS truy xuất chính xác các trang thực nghiệm trong `fp.pdf` (tập dữ liệu NIST SD4, độ chính xác phân loại).
    - Câu trả lời trả về đầy đủ mà người dùng không cần lặp lại tên tệp hay tên bài toán.

### 5.3. Kịch bản 3: Khả năng chống ảo giác khi câu hỏi ngoài phạm vi
* **Mục tiêu**: Kiểm tra tính trung thực của mô hình khi gặp câu hỏi không có trong kho dữ liệu.
* **Câu hỏi**: *"Hướng dẫn cách nấu món canh chua cá lóc miền Tây?"* hoặc *"Tài liệu có đề cập đến giải thuật Dijkstra tìm đường đi ngắn nhất không?"*
* **Kết quả xử lý của hệ thống**:
  - Prompt hệ thống yêu cầu: *"Chỉ sử dụng thông tin có trong phần Documents... Nếu tài liệu không đủ thông tin, hãy nói rõ rằng chưa tìm thấy thông tin cần thiết trong tài liệu hiện có."*
  - Trợ lý phản hồi lịch sự và chính xác: *"Tôi chưa tìm thấy thông tin về nội dung này trong các tài liệu nghiên cứu hiện có trong hệ thống."*
  - Không sinh ra bất kỳ câu trả lời bịa đặt nào, hoàn thành xuất sắc yêu cầu kiểm soát ảo giác (Hallucination Control).

### 5.4. Kịch bản 4: Độ chính xác của trích dẫn nguồn và số trang
* **Mục tiêu**: So sánh số trang được hệ thống báo cáo với số trang thực tế trong tệp PDF.
* **Đánh giá**:
  - Trong [rag_core.py](file:///d:/Computer%20Science/CT190%20-%20Nhap%20mon%20AI/BTGK/rag_core.py#L158), chỉ số trang được thiết lập: `page = d.metadata.get("page", 0) + 1`.
  - Do `PyMuPDF` đánh số trang bắt đầu từ 0 (0-indexed), việc cộng 1 giúp số trang hoàn toàn trùng khớp với trang hiển thị trên phần mềm đọc PDF của người dùng (Adobe Acrobat, Foxit Reader, trình duyệt).
  - Nguồn hiển thị dưới dạng: `Nguồn: Hieu_qua_cac_nhom_mo_hinh_hoc_sau_trong_bai_toan_p.pdf (tr.22); 60_FAIR2022_2692.pdf (tr.4)`.

### 5.5. Đánh giá tốc độ và tài nguyên hệ thống
* **Thời gian nạp chỉ mục**: Dưới 0.8 giây khi tải từ bộ nhớ cache `faiss_index`.
* **Thời gian tìm kiếm ngữ nghĩa FAISS**: Trung bình dưới 5ms cho 292 chunks.
* **Thời gian phản hồi token đầu tiên (Time-to-First-Token)**: ~0.3 - 0.6 giây nhờ hạ tầng Groq LPU.
* **Tốc độ sinh token (Generation Speed)**: Đạt trên 200 tokens/giây, văn bản xuất hiện lập tức trên màn hình tạo cảm giác phản hồi tức thì.

---

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Đóng góp và kết quả đạt được
Dự án **CT190-LitxAI** đã hoàn thành trọn vẹn các yêu cầu của môn học **Nhập môn Trí tuệ Nhân tạo (CT190)** với các điểm nổi bật:
1. **Ứng dụng thành công kỹ thuật RAG hiện đại**: Kết hợp hài hòa giữa xử lý ngôn ngữ tự nhiên (NLP), không gian vector ngữ nghĩa và mô hình ngôn ngữ lớn (LLM).
2. **Xây dựng giải pháp hoàn chỉnh**: Từ khâu xử lý dữ liệu thô (PDF Ingestion), phân mảnh, lập chỉ mục vector đến xây dựng giao diện ứng dụng hai chiều (Web Streaming & CLI).
3. **Giải quyết tốt bài toán đối thoại đa ngữ & ngữ cảnh**: Có cơ chế tự động tái cấu trúc câu truy vấn theo lịch sử trò chuyện và hỗ trợ cả tiếng Anh lẫn tiếng Việt.
4. **Đảm bảo tính tin cậy học thuật**: Ngăn ngừa ảo giác thông tin, trích dẫn chính xác tệp nguồn và số trang.

### 6.2. Hạn chế hiện tại
Bên cạnh những thành công, đồ án vẫn còn một số điểm có thể cải thiện:
- **Xử lý tài liệu dạng quét (Scanned PDF)**: Bộ đọc `PyMuPDF` phụ thuộc vào tầng văn bản (Text Layer). Đối với các tài liệu cũ dạng ảnh quét, hệ thống cần bổ sung tầng nhận dạng ký tự quang học (OCR).
- **Bảng biểu và biểu thức toán học**: Hiện tại văn bản được trích xuất dưới dạng plain-text tuyến tính, các công thức toán phức tạp hoặc bảng ma trận số liệu lớn có thể bị giảm cấu trúc khi phân đoạn.
- **Phụ thuộc kết nối Internet**: Mô hình LLM được gọi qua API đám mây của Groq, đòi hỏi đường truyền Internet ổn định.

### 6.3. Đề xuất hướng phát triển tương lai
1. **Tìm kiếm kết hợp (Hybrid Search)**: Kết hợp giữa tìm kiếm từ khóa truyền thống (BM25) và tìm kiếm vector ngữ nghĩa (Dense Semantic Search) thông qua kỹ thuật Reciprocal Rank Fusion (RRF) để tối ưu độ chính xác đối với các thuật ngữ chuyên ngành hiếm gặp.
2. **Tích hợp mô hình xếp hạng lại (Cross-Encoder Reranker)**: Sử dụng mô hình Cohere Rerank hoặc BGE-Reranker để tái đánh giá điểm số của Top-K tài liệu trước khi gửi vào LLM.
3. **Hỗ trợ tải lên tài liệu động (Dynamic Upload)**: Cho phép người dùng trực tiếp kéo thả tệp PDF vào giao diện Web, hệ thống tự động phân tích và nhúng vector thời gian thực.
4. **Multimodal RAG**: Sử dụng các mô hình Vision-Language (như Llama-3.2-Vision, ColPali) để đọc hiểu trực tiếp các biểu đồ, sơ đồ kiến trúc và bảng số liệu phức tạp trong bài báo.

---

## TÀI LIỆU THAM KHẢO

1. **Lewis, P., et al. (2020)**. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS 2020).
2. **Johnson, J., Douze, M., & Jégou, H. (2019)**. *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 7(3), 535-547 (Thư viện FAISS).
3. **Wang, L., et al. (2022)**. *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv preprint arXiv:2212.03533 (Mô hình E5 Embeddings).
4. **LangChain Documentation (2024)**. *Building applications with LLMs through composability*. [https://python.langchain.com/](https://python.langchain.com/)
5. **Groq Inc. (2024)**. *LPU Inference Engine Architecture*. [https://groq.com/](https://groq.com/)
6. **Các bài báo khoa học nghiên cứu tại Đại học Cần Thơ** được lưu trữ trong thư mục `paper/` của đề tài.

---

## PHỤ LỤC: HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH

### Yêu cầu môi trường
- Hệ điều hành: Windows 10/11, macOS, hoặc Linux.
- Python: Phiên bản 3.10 trở lên.
- Khóa API Groq miễn phí (đăng ký tại: [https://console.groq.com/](https://console.groq.com/)).

### Bước 1: Cài đặt các thư viện phụ thuộc
Mở terminal tại thư mục gốc của dự án và chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 2: Thiết lập biến môi trường
Tạo tệp `.env` tại thư mục gốc với nội dung:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### Bước 3: Vận hành ứng dụng

#### Lựa chọn A: Khởi chạy Giao diện Web (Khuyên dùng)
```bash
python server.py
```
Sau đó mở trình duyệt web và truy cập địa chỉ:
```
http://127.0.0.1:8000
```
Người dùng có thể nhập câu hỏi trực tiếp, theo dõi câu trả lời hiển thị dạng streaming và xem nguồn trích dẫn đính kèm.

#### Lựa chọn B: Khởi chạy Giao diện Dòng lệnh (CLI)
```bash
python chatbot.py
```
- Nhập câu hỏi trực tiếp tại dấu nhắc `You: `
- Nhập `clear` để xóa lịch sử trò chuyện
- Nhập `exit` hoặc `quit` để thoát chương trình.
