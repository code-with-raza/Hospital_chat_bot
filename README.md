# Northstar Hospital Chat Support

A local hospital assistant for doctor search, appointment booking, patient and appointment lookup, and hospital information. It does not provide diagnosis or treatment advice.

## Requirements

- Python 3.11+
- Node.js and npm
- An OpenAI-compatible API key for LLM features (optional for basic deterministic chat)

## Run locally (Windows PowerShell)

### 1. Start the backend

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your provider settings to `backend/.env` if using an LLM. Set `LLM_PROVIDER`, `LLM_API_KEY`, and `LLM_MODEL`. Keep real keys private; `.env` is ignored by Git.

Start the API from the `backend` directory:

```powershell
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`; health check: `http://localhost:8000/api/health`.

### 2. Start the frontend

In a second terminal, from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Data and notes

Doctor, patient, and appointment records are stored in JSON files under `backend/app/data/`. RAG source documents and local index files are managed by the backend. Conversation checkpoints are in memory and reset when the backend restarts.

This project is a local demo. JSON storage and patient lookup are not production-grade authentication or secure medical-record systems.
