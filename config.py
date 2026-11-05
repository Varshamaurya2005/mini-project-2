import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
STATIC_IMG_DIR = BASE_DIR / "static" / "images"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Aapka provide kiya gaya exact password
DB_PASSWORD = os.getenv("MYSQL_PASSWORD") or "priyashu@2005"

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": DB_PASSWORD,
    "database": os.getenv("MYSQL_DATABASE", "fake_news_research_db"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
}