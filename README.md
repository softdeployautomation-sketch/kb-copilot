# KB Copilot

Local **listen → transcribe → retrieve → reply** assistant for interview preparation and knowledge-base QA.

## Structure

```
kb-copilot/
├── task/                 # Core application code
│   ├── app.py            # Flask web server
│   ├── assist_core.py    # Core AI assist logic
│   ├── kb_store.py       # Knowledge base storage & retrieval
│   ├── retrieval.py      # BM25 retrieval logic
│   ├── ollama_client.py  # Ollama LLM client
│   ├── openai_compatible_client.py  # OpenAI-compatible LLM client
│   ├── transcription.py  # faster-whisper transcription
│   ├── templates_index.html  # Frontend UI
│   ├── run_server.sh     # Server startup script
│   └── start_kb_copilot_stack.sh  # Full stack startup
├── kb/                   # Knowledge base source files (.md, .txt)
├── .env                  # Local secrets (gitignored — never commit)
├── .env.example          # Example environment variables
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## What it does

KB Copilot captures audio from your microphone, transcribes it locally with **faster-whisper**, searches your knowledge base with **BM25**, and generates an answer using your configured LLM backend (Ollama or OpenAI-compatible).

- **Listen**: Record microphone audio, transcribe with Whisper, retrieve relevant KB chunks, generate an answer.
- **Assist (text)**: Type a question directly — same retrieval + generation pipeline without audio.
- **Channelry AI**: Optionally route through Channelry's external AI chat endpoint.

## Prerequisites

- **Python 3.9+**
- **ffmpeg** on the PATH (Whisper decodes browser WebM/MP4): `brew install ffmpeg`
- **Ollama** desktop or CLI (optional, if using `KB_LLM_BACKEND=ollama`), or an **OpenAI-compatible API key** (Groq, OpenRouter, etc.)

## Quick start

```bash
cd kb-copilot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# Start the server
./task/run_server.sh
# or: python task/app.py
```

Open `http://127.0.0.1:5055` in your browser.

## Environment variables

Copy `.env.example` to `.env` and fill in your values. `.env` is gitignored — never commit real keys.

| Variable | Default | Meaning |
|----------|---------|---------|
| `KB_ROOT` | `./kb` | Directory to index for knowledge base |
| `KB_LLM_BACKEND` | `openai_compatible` | LLM backend: `ollama` or `openai_compatible` |
| `OPENAI_COMPAT_API_KEY` | *(required)* | API key for OpenAI-compatible endpoint |
| `OPENAI_COMPAT_BASE_URL` | `https://api.groq.com/openai/v1` | Base URL for OpenAI-compatible API |
| `OPENAI_COMPAT_QUICK_MODEL` | `llama-3.1-8b-instant` | Fast model for quick answers |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama base URL (when using Ollama backend) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `WHISPER_MODEL` | `base` | Whisper size: `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `PORT` | `5055` | Web UI port |
| `MAX_UPLOAD_MB` | `32` | Max recorded clip size in MB |

## API endpoints

- `POST /api/listen_assist` — multipart form: `audio` blob + options (`session_brief`, `top_k`, `use_ollama`, `spoken`, `allow_general`, `language`).
- `POST /api/transcribe` — audio only → `{ text, meta }`.
- `POST /api/assist` — JSON `question` + options → generated answer.

## Ethics and policy

Use only where allowed by the organizer, employer, or institution. Do not use realtime assistance where it violates rules or expectations; you are responsible for disclosure and honesty.

## License

MIT.
