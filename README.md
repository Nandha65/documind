# DocuMind: Multi-Provider Local RAG Assistant

DocuMind is an end-to-end Retrieval-Augmented Generation (RAG) system built with **LlamaIndex**, **ChromaDB**, and **Streamlit**. It enables users to index and chat with local unstructured documents (PDFs, text files) with support for switching dynamically between **Google Gemini** and **OpenAI**.

---

## Key Features

- **Dual LLM & Embedding Support:** Switch dynamically between Google Gemini (`gemini-3.6-flash` / `gemini-embedding-2-preview`) and OpenAI (`gpt-4o-mini` / `text-embedding-3-small`) directly from the sidebar.
- **Isolated Vector Collections:** Employs dedicated ChromaDB collections (`documind_gemini` and `documind_openai`) to handle different embedding vector dimensions without collisions or indexing errors.
- **Local-First & Zero-Config:** In-process vector storage via ChromaDB requires no external database servers or cloud cluster setups.
- **Interactive Chat Interface:** Clean chat interface built with Streamlit.

---

## Tech Stack & Architecture

| Layer | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Industry standard for AI engineering and rapid prototyping. |
| **Orchestration** | LlamaIndex | Handles document ingestion, chunking, and similarity retrieval cleanly without boilerplate. |
| **Vector Store** | ChromaDB (Local) | Embedded vector store running locally in-process; keeps data private and repository clone-and-run friendly. |
| **LLMs & Embeddings** | Google Gemini & OpenAI | Provides flexibility between Google AI Studio's free tier and OpenAI models. |
| **UI Framework** | Streamlit | Lightweight Python-native framework for real-time chat interfaces. |

---

## Project Structure

```text
documind/
├── data/                  # Source documents (.txt, .pdf)
├── src/
│   ├── __init__.py
│   └── pipeline.py        # Dual-provider ingestion, embedding, and query logic
├── app.py                 # Streamlit multi-provider chat UI
├── .env.example           # Template for environment variables
├── .gitignore             # Git exclusion rules
├── requirements.txt       # Project dependencies
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

Open `.env` and add your active keys (you can provide one or both):
```env
GOOGLE_API_KEY=AIzaSy...your_gemini_key_here
OPENAI_API_KEY=sk-proj-...your_openai_key_here
```

### 5. Add Source Documents
Place any `.txt` or `.pdf` files you wish to index inside the `data/` directory.

The index is built on first run and reused afterwards. If you add or change documents later, delete the `chroma_db/` folder so the index is rebuilt.

### 6. Run the Application
Start the Streamlit web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, pick your desired provider from the sidebar, and begin querying your documents!

---

## License
MIT License. Feel free to use and modify for personal or commercial projects.
