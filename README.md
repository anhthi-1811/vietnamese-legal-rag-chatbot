# Vietnamese Legal AI Chatbot

A multi-turn Vietnamese legal question-answering system built with **Retrieval-Augmented Generation (RAG)**, **BM25**, **BGE-M3**, **FAISS**, **BGE-Reranker-v2-m3**, and an LLM-based answer generator.

The system retrieves relevant legal evidence from a corpus of **261,597 Vietnamese legal documents**, reranks candidate documents, generates evidence-grounded answers with document-level citations, and supports conversational follow-up questions through query rewriting and short-term conversation history.

---

## 1. Overview

Legal question answering requires more than generating fluent text. A reliable system must retrieve the right legal evidence, preserve the scope of the user's question, avoid unsupported claims, and provide citations that can be inspected.

This project implements an end-to-end Vietnamese legal RAG pipeline with a conversational interface:

```text
User Message
    ↓
Conversation History
    ↓
Query Rewriter
    ↓
Standalone Legal Question
    ↓
BM25 Top-50 + BGE-M3 Top-50
    ↓
Candidate Union
    ↓
BGE-Reranker-v2-m3
    ↓
Top-5 Legal Evidence
    ↓
Passage Selection for Long Documents
    ↓
Context Builder
    ↓
LLM Answer Generator
    ↓
Answer + Legal Citations
    ↓
Conversation History
```

The project focuses on three goals:

1. **High-quality legal retrieval**
2. **Evidence-grounded answer generation**
3. **Conversational follow-up support**

---

## 2. Main Features

- Multi-turn Vietnamese legal chatbot
- Conversational follow-up question handling
- Query rewriting for context-dependent questions
- Hybrid lexical + dense retrieval
- BM25 lexical search
- BGE-M3 dense embeddings
- FAISS vector search
- Cross-encoder reranking with BGE-Reranker-v2-m3
- Passage selection for long legal documents
- Evidence-grounded LLM answer generation
- Document-level legal citations
- Deterministic citation validation
- Retrieval and RAG evaluation pipeline
- Interactive CLI
- Streamlit web interface
- Per-stage latency logging for debugging and profiling

---

## 3. Dataset

This project uses the **YuITC Vietnamese Legal Documents** dataset.

### Corpus

- **261,597 legal documents**
- Each document is identified by a unique `cid`
- Documents contain Vietnamese legal text
- Retrieval and evaluation are performed at the CID/document level

### Corpus characteristics

The corpus is mostly composed of short-to-medium legal passages, but also contains a small number of very long documents.

Observed token statistics using the BGE-M3 tokenizer:

| Statistic | Tokens |
|---|---:|
| Median | 204 |
| Mean | 283.96 |
| P90 | 636 |
| P95 | 728 |
| P99 | 1,034 |
| Maximum | 67,371 |

Because benchmark labels are document-level CIDs and most documents are already relatively short, the system retrieves documents directly instead of globally chunking the entire corpus.

Long documents are handled later with local passage selection after reranking.

---

## 4. Data Preprocessing

Preprocessing is intentionally conservative because Vietnamese legal text contains important structural information such as:

- Article numbers
- Clause numbers
- Legal identifiers
- Dates
- Punctuation
- Line breaks
- Formal legal terminology

The preprocessing pipeline:

- normalizes Unicode to NFC
- removes anomalous whitespace and HTML artifacts
- preserves legal numbering and punctuation
- preserves document-level CID identity
- retains short documents instead of filtering them aggressively
- flags long documents for later passage selection

Processed corpus:

```text
data/processed/documents_clean.parquet
```

---

## 5. Retrieval Architecture

### 5.1 BM25 Lexical Retrieval

BM25 provides lexical matching and is especially useful for legal queries containing:

- article numbers
- clause numbers
- named laws
- exact legal phrases
- specific terminology

No aggressive stopword removal or stemming is used.

### 5.2 BGE-M3 Dense Retrieval

The dense retriever uses:

```text
BAAI/bge-m3
```

Document embeddings are precomputed and stored for vector search using FAISS.

Dense retrieval is particularly useful for semantic matching when the question and legal document use different wording.

### 5.3 Hybrid Candidate Generation

The final retrieval strategy does not directly fuse BM25 and dense scores.

Instead:

```text
BM25 Top-50
+
BGE-M3 Top-50
↓
Deduplicated Candidate Union
```

This strategy was selected because BM25 and BGE-M3 retrieve complementary evidence.

### Candidate recall

| Candidate depth | BM25 Recall | BGE-M3 Recall | Union Recall | Avg. Pool Size |
|---|---:|---:|---:|---:|
| 20 | 0.6802 | 0.8157 | 0.8560 | 33.4 |
| 50 | 0.7649 | 0.8783 | **0.9073** | 83.9 |
| 100 | 0.8180 | 0.9128 | 0.9355 | 168.4 |

Top-50 from each retriever was selected as the final trade-off between candidate coverage and reranking cost.

---

## 6. Retrieval Baselines

### BM25

| Metric | Score |
|---|---:|
| Recall@1 | 0.2564 |
| Recall@5 | 0.4911 |
| Recall@10 | 0.5798 |
| Recall@20 | 0.6578 |
| MRR@10 | 0.3713 |
| nDCG@10 | 0.4149 |

### BGE-M3

| Metric | Score |
|---|---:|
| Recall@1 | 0.3717 |
| Recall@5 | 0.6436 |
| Recall@10 | 0.7276 |
| Recall@20 | 0.7921 |
| MRR@10 | 0.5079 |
| nDCG@10 | 0.5529 |

BGE-M3 substantially outperformed BM25 overall, but BM25 still recovered relevant documents missed by dense retrieval.

### Complementarity at Top-20

| Outcome | Percentage |
|---|---:|
| Both correct | 63.99% |
| BM25 only | 4.03% |
| BGE-M3 only | 17.58% |
| Both wrong | 14.40% |

This complementarity motivated the candidate-union design.

---

## 7. RRF Experiment

Reciprocal Rank Fusion (RRF) was evaluated as an alternative hybrid strategy.

The best configuration used:

```text
rrf_k = 10
```

However, shallow RRF rankings did not consistently outperform BGE-M3 alone.

| Metric | BGE-M3 | RRF |
|---|---:|---:|
| Recall@10 | 0.7276 | 0.7225 |
| Recall@20 | 0.7921 | 0.7927 |
| MRR@10 | 0.5079 | 0.4786 |
| nDCG@10 | 0.5529 | 0.5297 |

RRF is therefore kept as an experiment rather than the final retrieval architecture.

---

## 8. Cross-Encoder Reranking

The final candidate pool is reranked using:

```text
BAAI/bge-reranker-v2-m3
```

The reranker jointly scores each `(question, candidate document)` pair and returns the final ranking.

### Reranker results

| Metric | Score |
|---|---:|
| Recall@1 | 0.4627 |
| Recall@5 | 0.7372 |
| Recall@10 | **0.8023** |
| Recall@20 | 0.8465 |
| MRR@10 | **0.6023** |
| nDCG@10 | **0.6421** |

The final RAG system uses the top 5 reranked documents as evidence candidates.

---

## 9. Passage Selection

Global corpus chunking is not used.

Instead, passage selection is applied only after reranking and only when a retrieved legal document is long.

```text
Retrieved Document
    ↓
Is the document long?
    ├── No  → use the document directly
    └── Yes → select the most relevant passages
```

This preserves document-level benchmark alignment while reducing unnecessary context length.

---

## 10. Conversational RAG

The chatbot supports follow-up questions that depend on previous turns.

Example:

```text
User:
Tội giết người bị phạt thế nào?

Assistant:
...

User:
Nếu không thuộc khoản 1 thì sao?
```

The second question is not sufficiently clear when retrieved independently.

The system therefore uses a query-rewriting stage:

```text
Conversation History
+
Current User Question
↓
Query Rewriter
↓
Standalone Legal Question
```

Example rewritten query:

```text
Người phạm tội giết người không thuộc các trường hợp
quy định tại khoản 1 Điều 123 thì bị phạt như thế nào?
```

The rewritten standalone question is then used for BM25 and dense retrieval.

### Short-term conversation memory

The chatbot stores recent user and assistant turns within the current session.

This is used only to resolve conversational references.

Conversation history is **not treated as legal evidence**.

All legal claims in generated answers must be supported by retrieved legal documents.

---

## 11. Query Rewriting

The query rewriter supports three outcomes:

```text
unchanged
rewritten
needs_clarification
```

### `unchanged`

The user question is already self-contained.

Example:

```text
Tội trộm cắp tài sản bị phạt như thế nào?
```

### `rewritten`

The question depends on previous conversation context.

Example:

```text
Nếu không thuộc khoản 1 thì sao?
```

### `needs_clarification`

The reference cannot be safely resolved from the conversation.

Example:

```text
Cái đó thì sao?
```

If the context is insufficient, the chatbot asks the user to clarify instead of guessing.

---

## 12. Answer Generation

The generator receives:

- the legal question
- retrieved evidence
- allowed document CIDs

The generator returns structured output:

```json
{
  "answer": "...",
  "cited_cids": ["..."],
  "answer_status": "answered"
}
```

Possible answer states include:

```text
answered
insufficient_evidence
```

The conversational layer can additionally handle:

```text
needs_clarification
```

The generator is instructed to:

- answer only from retrieved evidence
- preserve the requested legal scope
- avoid unsupported legal inference
- prefer specific evidence over broad evidence
- cite supporting legal documents
- abstain when evidence is insufficient
- avoid treating chat history as legal evidence

---

## 13. Citation Validation

The project includes a deterministic citation checker.

It verifies:

- whether cited CIDs exist in the provided context
- whether the answer contains undeclared citations
- whether declared citations are actually used
- whether an answered response contains citations when required
- whether citation structure is internally consistent

Citation structure validation is separate from semantic citation support.

A structurally valid citation can still fail to support the claim semantically, so citation support is also evaluated separately.

---

## 14. RAG Evaluation

The final generator was evaluated on a fixed held-out subset of:

```text
250 test queries
```

The final generator configuration was frozen before this evaluation.

### Final test results

| Metric | Result |
|---|---:|
| Answered | 238 / 250 |
| Answer rate | 95.2% |
| Correctness | **90.8%** |
| Completeness | **90.8%** |
| Fully grounded | **97.5%** |
| Citation supported | **98.3%** |
| Mean correctness | 1.85 / 2 |
| Mean completeness | 1.86 / 2 |
| Mean groundedness | 1.96 / 2 |
| Mean citation support | 1.97 / 2 |

### Abstention

| Metric | Result |
|---|---:|
| Insufficient-evidence responses | 12 |
| Appropriate abstentions | 5 |
| Unnecessary abstentions | 7 |

The main remaining weakness is therefore not unsupported generation, but overly conservative behavior in some cases.

---

## 15. Semantic Evaluation Definitions

### Correctness

Measures whether the generated answer is correct relative to benchmark/reference evidence.

### Completeness

Measures whether the answer includes the necessary information required to answer the question.

### Groundedness

Measures whether claims in the generated answer are supported by the evidence actually retrieved and shown to the generator.

### Citation Support

Measures whether the cited legal documents directly support the claims associated with them.

These dimensions are evaluated separately because an answer can be:

- grounded but incorrect relative to the benchmark
- correct but incompletely supported
- structurally cited but semantically unsupported

---

## 16. Error Analysis

Final test error tags included:

| Error Type | Count |
|---|---:|
| Generator omission | 13 |
| Unnecessary abstention | 7 |
| Valid abstention | 5 |
| Conflicting evidence | 5 |
| Evidence selection error | 4 |
| Legal inference error | 3 |
| Temporal legal conflict | 2 |
| Wrong scope | 2 |
| Unsupported claim | 2 |
| Incomplete information | 1 |
| Retrieval miss | 1 |
| Equivalent non-gold evidence | 1 |

The final system's most important remaining issues are:

- missing details despite sufficient evidence
- unnecessary abstention
- conflicting legal evidence
- evidence selection
- temporal legal conflicts

---

## 17. Web Interface

The project includes a Streamlit-based web chatbot.

Main interface features:

- multi-turn conversation
- legal question input
- generated answer display
- legal document citations
- retrieved evidence inspection
- conversation reset
- optional debugging information such as rewritten queries and retrieved documents

Run the web app with:

```bash
streamlit run app/streamlit_app.py
```

---

## 18. CLI Usage

### Single-turn RAG

```bash
python app/cli_app.py
```

### Multi-turn chatbot

```bash
python app/cli_chatbot.py
```

The chatbot can support commands such as:

```text
exit
new
history
```

---

## 19. Project Structure

```text
Vietnamese-Legal-RAG/
│
├── app/
│   ├── cli_app.py
│   ├── cli_chatbot.py
│   └── streamlit_app.py
│
├── artifacts/
│   ├── bm25/
│   └── dense/
│
├── configs/
│   ├── retrieval.yaml
│   ├── rag.yaml
│   └── chatbot.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── documents_clean.parquet
│   └── benchmark/
│
├── experiments/
│   ├── logs/
│   ├── figures/
│   └── results/
│
├── notebooks/
│   ├── 01_eda_corpus.ipynb
│   ├── 02_eda_benchmark.ipynb
│   ├── 03_retrieval_analysis.ipynb
│   ├── 04_error_analysis.ipynb
│   └── 05_chatbot_analysis.ipynb
│
├── scripts/
│   ├── profile_chatbot.py
│   └── evaluate_multiturn.py
│
├── src/
│   ├── data/
│   │   └── cleaner.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── faiss_index.py
│   │   └── reranker.py
│   ├── rag/
│   │   ├── pipeline.py
│   │   ├── passage_retriever.py
│   │   ├── context_builder.py
│   │   ├── generator.py
│   │   └── citation_checker.py
│   ├── chat/
│   │   ├── session.py
│   │   ├── query_rewriter.py
│   │   └── chatbot.py
│   └── evaluation/
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

The exact local structure may vary slightly depending on where evaluation scripts and experiment files are stored.

---

## 20. Installation

### 20.1 Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Vietnamese-Legal-RAG
```

### 20.2 Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 20.3 Install dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
pandas
numpy
pyarrow
torch
transformers
FlagEmbedding
faiss-cpu
bm25s
google-genai
python-dotenv
PyYAML
streamlit
pytest
```

Use the appropriate FAISS and PyTorch build for your operating system and GPU environment.

---

## 21. Environment Variables

Create `.env` from `.env.example`.

Example:

```env
GOOGLE_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

---

## 22. Running with Docker

The project can also be run inside a Docker container to provide a reproducible environment without requiring users to manually install Python dependencies.

### 1. Build the Docker image

From the project root directory:

```bash
docker build -t vietnamese-legal-chatbot .
```

This command builds a Docker image containing the application source code and all required Python dependencies defined in `requirements.txt`.

---

### 2. Prepare environment variables

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_api_key_here
```

The `.env` file is used to provide API credentials to the application at runtime.

For security reasons, `.env` should never be committed to GitHub.

An example configuration can be provided through:

```text
.env.example
```

with content such as:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

### 3. Run the Streamlit chatbot

If the required retrieval artifacts and processed corpus are already included in the expected project directories, run:

```bash
docker run --rm \
  --env-file .env \
  -p 8501:8501 \
  vietnamese-legal-chatbot
```

Then open the application in your browser:

```text
http://localhost:8501
```

On Windows PowerShell:

```powershell
docker run --rm `
  --env-file .env `
  -p 8501:8501 `
  vietnamese-legal-chatbot
```

---

### 4. Run with external retrieval artifacts

Large retrieval artifacts and processed datasets are not recommended to be included directly inside the Docker image.

Instead, they can be mounted from the host machine at runtime.

The application expects directories such as:

```text
artifacts/
├── bm25/
└── dense/

data/
└── processed/
    └── documents_clean.parquet
```

Run the container with mounted directories:

```bash
docker run --rm \
  --env-file .env \
  -p 8501:8501 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/data/processed:/app/data/processed" \
  vietnamese-legal-chatbot
```

On Windows PowerShell:

```powershell
docker run --rm `
  --env-file .env `
  -p 8501:8501 `
  -v "${PWD}/artifacts:/app/artifacts" `
  -v "${PWD}/data/processed:/app/data/processed" `
  vietnamese-legal-chatbot
```

This keeps large indexes and datasets outside the Docker image while allowing the application to access them inside the container.

---

### 5. Run with Docker Compose

If `docker-compose.yml` is available in the repository, the complete application can be started with:

```bash
docker compose up --build
```

The Streamlit interface will be available at:

```text
http://localhost:8501
```

To stop the application:

```bash
docker compose down
```

A typical `docker-compose.yml` mounts the retrieval artifacts and processed data automatically:

```yaml
services:
  legal-chatbot:
    build: .
    container_name: vietnamese-legal-chatbot

    env_file:
      - .env

    ports:
      - "8501:8501"

    volumes:
      - ./artifacts:/app/artifacts
      - ./data/processed:/app/data/processed

    restart: unless-stopped
```

---

### 6. Docker image configuration

A minimal `Dockerfile` for the Streamlit application can be defined as:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

The Docker image contains:

```text
Python runtime
Application source code
Python dependencies
Streamlit application
RAG pipeline code
```

Large retrieval artifacts and processed datasets can remain on the host machine and be mounted into the container at runtime.

---

### 7. Docker ignore configuration

A `.dockerignore` file is recommended to prevent unnecessary files and large artifacts from being copied into the Docker image.

Example:

```text
.git
.github

.env
.venv
__pycache__
*.pyc
.pytest_cache

notebooks/
experiments/

data/raw/
data/processed/

artifacts/
```

Because `artifacts/` and `data/processed/` are excluded from the image, they should be mounted into the container when the application starts.

---

### 8. CPU and GPU inference

BGE-M3 and BGE-Reranker-v2-m3 are loaded when the application starts.

Running the cross-encoder reranker on CPU can be significantly slower than running it on a GPU.

The default Docker configuration is suitable for reproducibility and local demonstration.

For GPU-accelerated inference, the host system requires:

```text
NVIDIA GPU
Compatible NVIDIA driver
NVIDIA Container Toolkit
CUDA-compatible PyTorch installation
```

GPU execution can then be enabled using Docker's GPU support, depending on the local environment.

---

## 23. Required Artifacts

The application expects prebuilt retrieval artifacts before running the chatbot.

Typical locations are:

```text
artifacts/
├── bm25/
└── dense/
```

The BM25 directory contains the lexical retrieval index.

The dense retrieval directory contains the FAISS index and the corresponding CID mapping used to map retrieved vectors back to legal documents.

The application also expects the processed legal corpus:

```text
data/
└── processed/
    └── documents_clean.parquet
```

These files are loaded when the RAG pipeline starts.

Large retrieval artifacts, embeddings, FAISS indexes, and processed datasets should generally not be committed directly to Git because they can be significantly larger than the source code.

Instead, they can be:

- generated locally using the project preprocessing and indexing scripts;
- stored separately from the Git repository;
- mounted into Docker containers at runtime;
- or distributed through an external storage service.

The expected runtime structure is:

```text
Vietnamese-Legal-RAG/
│
├── artifacts/
│   ├── bm25/
│   └── dense/
│
├── data/
│   └── processed/
│       └── documents_clean.parquet
│
├── app/
├── configs/
├── src/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

The exact artifact filenames inside each directory depend on the implementation used to save the BM25 and FAISS indexes.

---

## 23. Configuration

Retrieval parameters are stored in:

```text
configs/retrieval.yaml
```

RAG generation and context parameters are stored in:

```text
configs/rag.yaml
```

Conversational settings are stored in:

```text
configs/chatbot.yaml
```

Typical retrieval configuration:

```yaml
bm25:
  top_k: 50

dense:
  model_name: BAAI/bge-m3
  top_k: 50
  use_fp16: true

reranker:
  model_name: BAAI/bge-reranker-v2-m3
  max_length: 512
```

The exact values should match the final configuration used in the repository.

---

## 24. Running the System

### Start the Streamlit chatbot

```bash
streamlit run app/streamlit_app.py
```

### Run the interactive CLI chatbot

```bash
python app/cli_chatbot.py
```

### Run the original single-turn pipeline

```bash
python app/cli_app.py
```

---

## 25. Evaluation Workflow

The evaluation workflow is separated into two levels.

### Retrieval evaluation

Evaluates:

```text
BM25
BGE-M3
Hybrid candidate union
Cross-encoder reranker
```

using metrics such as:

```text
Recall@K
MRR@K
nDCG@K
```

### RAG evaluation

Evaluates generated answers using:

```text
Correctness
Completeness
Groundedness
Citation Support
Abstention Appropriateness
```

The final test set is treated as held-out evaluation data and should not be used for further tuning.

---

## 26. Design Decisions

### Why no global chunking?

The benchmark is document/CID-level, and most legal records are already relatively short.

Global chunking would complicate alignment between retrieved chunks and benchmark labels.

Instead, passage selection is applied only to long reranked documents.

### Why keep BM25 if BGE-M3 is stronger?

BM25 still retrieves relevant documents that dense search misses.

The candidate-union experiment showed that combining both retrievers significantly increases candidate coverage.

### Why use a cross-encoder reranker?

Dense retrieval is fast but scores query and document embeddings independently.

The reranker jointly reads the question and document, allowing more accurate relevance scoring.

### Why use query rewriting for chatbot follow-ups?

Follow-up questions often contain references such as:

```text
"khoản đó"
"trường hợp trên"
"nếu không thuộc khoản 1"
```

These expressions are difficult to retrieve correctly without conversational context.

Query rewriting converts them into standalone legal questions before retrieval.

### Why not use long-term memory?

The main purpose of memory in this system is to understand recent conversational references.

Long-term user-profile memory is not required for reliable legal retrieval and would add unnecessary complexity.

---

## 27. Limitations

### Temporal legal validity

Retrieved legal evidence may contain provisions from different time periods.

A response can be grounded in retrieved evidence while still reflecting outdated or conflicting legal text.

Future versions should incorporate explicit legal validity metadata where available.

### Conflicting evidence

Multiple retrieved legal documents can sometimes contain conflicting provisions.

The system can identify some of these cases, but conflict resolution remains a challenging legal reasoning problem.

### Unnecessary abstention

The generator can occasionally refuse to answer even when retrieved evidence is sufficient.

### Retrieval dependency

The generator cannot recover legal evidence that is missing from the candidate pool.

Retrieval quality therefore remains a critical upper bound for downstream answer quality.

### Latency

Cross-encoder reranking can be expensive on CPU.

GPU inference is strongly recommended for interactive use.

### Legal disclaimer

This project is an AI research and engineering system and is **not a substitute for professional legal advice**.

---

## 28. Future Improvements

Potential improvements include:

- temporal legal validity awareness
- conflict-aware evidence handling
- faster reranker inference
- query and retrieval caching
- improved abstention calibration
- broader multi-turn evaluation
- document metadata display
- stronger evidence highlighting in the web interface
- Docker deployment
- API deployment with FastAPI
- production logging and monitoring

---

## 29. Key Results

### Retrieval

```text
Candidate Union Recall@50: 90.73%
Reranker Recall@10:        80.23%
Reranker MRR@10:           60.23%
Reranker nDCG@10:          64.21%
```

### End-to-End RAG

```text
Answer Correctness:  90.8%
Completeness:        90.8%
Groundedness:        97.5%
Citation Support:    98.3%
Answer Rate:         95.2%
```

---

## 30. Summary

This project demonstrates a complete Vietnamese legal AI system covering:

```text
Data preprocessing
→ Hybrid retrieval
→ Dense vector search
→ Candidate generation
→ Cross-encoder reranking
→ Passage selection
→ RAG generation
→ Citation validation
→ Semantic evaluation
→ Multi-turn conversation
→ Web deployment
```

Rather than treating the project only as a chatbot interface, the system emphasizes measurable retrieval quality, evidence-grounded generation, citation reliability, and transparent evaluation.

---

## Disclaimer

This project is intended for educational, research, and portfolio purposes.

Generated responses may contain errors, outdated legal information, or incomplete interpretations. Users should verify important legal information against authoritative and currently effective legal sources and consult qualified legal professionals when necessary.
