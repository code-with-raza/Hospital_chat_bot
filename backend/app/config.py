import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
load_dotenv(BASE_DIR.parent / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

DEFAULT_BASE_URLS = {
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
}
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URLS.get(LLM_PROVIDER)
