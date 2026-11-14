# DocuMind: Multi-Provider Local RAG Assistant

DocuMind is an end-to-end Retrieval-Augmented Generation (RAG) system built with **LlamaIndex**, **ChromaDB**, and **Streamlit**. It enables users to index and chat with local unstructured documents (PDFs, text files) with support for switching dynamically between **Google Gemini**, **OpenAI** and **Anthropic Claude**.

---

## Key Features

- **Multi-Provider LLM & Embedding Support:** Switch dynamically between Google Gemini (`gemini-3.6-flash` / `gemini-embedding-2-preview`), OpenAI (`gpt-4o-mini` / `text-embedding-3-small`) and Anthropic Claude (`claude-opus-5-5` / local `BAAI/bge-small-en-v1.5` embeddings) directly from the sidebar.
- **Streaming Answers:** Responses appear token by token as they are generated.
- **Follow-up Questions:** The recent conversation is sent with each question, so follow-ups like "who do I ask about it?" work.
- **Sources Under Every Answer:** A collapsible panel lists the documents (and PDF pages) each answer was drawn from, with an excerpt.
- **Automatic Re-indexing:** Add, edit or remove files in `data/` and the index is rebuilt on the next question, without restarting the app.
- **PDF, Markdown and Text Documents:** PDFs are parsed with `pypdf`; each page is indexed separately.
- **Isolated Vector Collections:** Employs dedicated ChromaDB collections (`documind_gemini`, `documind_openai` and `documind_claude`) to handle different embedding vector dimensions without collisions or indexing errors.
- **Local-First & Zero-Config:** In-process vector storage via ChromaDB requires no external database servers or cloud cluster setups.
- **Interactive Chat Interface:** Clean chat interface built with Streamlit.

---

## Tech Stack & Architecture

| Layer | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Industry standard for AI engineering and rapid prototyping. |
| **Orchestration** | LlamaIndex | Handles document ingestion, chunking, and similarity retrieval cleanly without boilerplate. |
| **Vector Store** | ChromaDB (Local) | Embedded vector store running locally in-process; keeps data private and repository clone-and-run friendly. |
| **LLMs & Embeddings** | Google Gemini, OpenAI & Anthropic Claude | Provides flexibility between Google AI Studio's free tier, OpenAI and Claude models. |
| **Local Embeddings** | FastEmbed | Embeds documents and questions on-device for the Claude provider, since Anthropic has no embeddings API. |
| **UI Framework** | Streamlit | Lightweight Python-native framework for real-time chat interfaces. |

---

## How It Works

DocuMind works in two stages. On startup it builds (or reuses) a vector index of your documents for the selected provider; the index is cached and rebuilt only when the files in `data/` change. Then, for every question, it finds the most relevant chunks and asks the LLM to answer from them, with the recent conversation for context.

```mermaid
flowchart TD
    subgraph setup["Stage 1: Startup"]
        A["Pick a provider in the sidebar"] --> B["configure_provider()<br/>LLM + embedding model + Chroma collection"]
        B --> C{"Index built from the<br/>current data/ files?"}
        C -- "No (first run or files changed)" --> D["Load files from data/"]
        D --> E["Split into chunks"]
        E --> F["Embed each chunk"]
        F --> G[("chroma_db/")]
        C -- "Yes" --> G
    end

    subgraph ask["Stage 2: Every question"]
        H["User asks a question"] --> I["Embed the previous and new question<br/>API call for Gemini and OpenAI, local for Claude"]
        I --> J["Retrieve the 4 most similar chunks"]
        J --> K["Build prompt: chunks + recent chat + question"]
        K --> L["LLM writes the answer"]
        L --> M["Stream tokens into the chat<br/>and list the sources"]
    end

    G -. "similarity search" .-> J
```

| Provider | LLM | Embeddings | Chroma collection |
| :--- | :--- | :--- | :--- |
| Google Gemini | `gemini-3.6-flash` (thinking level: low) | `gemini-embedding-2-preview` (API) | `documind_gemini` |
| OpenAI | `gpt-4o-mini` | `text-embedding-3-small` (API) | `documind_openai` |
| Anthropic Claude | `claude-opus-5-5` (effort: low, server-side refusal fallback) | `BAAI/bge-small-en-v1.5` (local, via FastEmbed) | `documind_claude` |

Each provider gets its own collection because each embedding model produces vectors of a different size. Response time is dominated by the two network calls in stage 2: embedding the question and generating the answer. The Claude provider embeds locally, so it only makes the second one.

For follow-up questions, the search combines the previous question with the new one. LlamaIndex's default is to ask the LLM to rewrite each follow-up as a standalone question first, but that adds an LLM call to every question after the first.

---

## Project Structure

```text
documind/
├── data/                  # Source documents (.txt, .md, .pdf); ships with fictional sample data
├── src/
│   ├── __init__.py
│   ├── pipeline.py        # Ingestion, re-indexing, retrieval and chat logic
│   ├── claude_llm.py      # LlamaIndex LLM that calls Claude via the Anthropic SDK
│   └── errors.py          # Friendly messages for provider errors
├── tests/                 # Offline tests (no API keys or network needed)
├── app.py                 # Streamlit multi-provider chat UI
├── .env.example           # Template for environment variables
├── .gitignore             # Git exclusion rules
├── requirements.txt       # Project dependencies (pinned)
├── requirements-dev.txt   # Adds pytest for running the tests
├── pytest.ini             # Test configuration
└── README.md              # Project documentation
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/documind.git
cd documind
```

### 2. Set Up Virtual Environment

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy `.env.example` to create your local `.env` file:

**Windows:**
```cmd
copy .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Open `.env` and add a key for each provider you plan to use:
```env
GOOGLE_API_KEY=AIzaSy...your_gemini_key_here
OPENAI_API_KEY=sk-proj-...your_openai_key_here
ANTHROPIC_API_KEY=sk-ant-...your_anthropic_key_here
```

The first time you use the Claude provider, the local embedding model (~70 MB) is downloaded into `.fastembed_cache/`.

### 5. Add Source Documents
Place any `.txt`, `.md` or `.pdf` files you wish to index inside the `data/` directory. The repository ships with fictional sample documents for a made-up company, Lumenfield Labs, so you can try the app straight away; remove them when you add your own.

The index is built on first run and reused afterwards. When you add, edit or remove files in `data/`, it is rebuilt automatically on the next question.

### 6. Run the Application
Start the Streamlit web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, pick your desired provider from the sidebar, and begin querying your documents! With the sample data, try:

- How many days of annual leave do employees get?
- What is the daily meal allowance when travelling? Then ask: *Who do I contact with questions about it?*
- How many paying customers is Ledgerly aiming for by March 2027? (answered from the PDF)

---

## Running the Tests

The tests run offline: they use fake embeddings, a fake Anthropic API and Streamlit's `AppTest`, so no API keys are needed.

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Production Starter Kits

Building a commercial product? **[PlanForge](https://planforge.web.app)** is a TypeScript engine for
spatial layout editors and interactive grid planners: drag-and-drop layouts, snapping, live pricing and print-ready
vector PDF export.

- [Try the live demo](https://planforge.web.app): six industry planners on one engine
- [Get the source code](https://nandha4.gumroad.com/l/planforge) ($49 one-time), or
  [sponsor on GitHub](https://github.com/sponsors/Nandha65) ($29/month) for repo access and every update

---

## License
MIT License. Feel free to use and modify for personal or commercial projects.
