import os
import wave
import json
import shutil
import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import clip
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import pdfplumber
from docx import Document
import pandas as pd
from sentence_transformers import SentenceTransformer
import sqlite3
from dotenv import load_dotenv
import logging
import mimetypes
from pathlib import Path
import hashlib
from contextlib import asynccontextmanager

# Setup logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("media-indexer")

# FastAPI app
app = FastAPI(title="Media Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurations
device = "cpu"
DB_PATH = os.getenv("DB_PATH", "/tmp/media_index.db")
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/tmp/vosk-model-small-en-us-0.15")  # Changed to /tmp
TRANSCRIPTIONS_FOLDER = "/tmp/transcriptions"
MEADOCS_FOLDER = "/tmp/Meadocs_data"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 20))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)
os.makedirs(MEADOCS_FOLDER, exist_ok=True)

# Supported media types
SUPPORTED_MEDIA = {
    "photo": {".jpg", ".jpeg", ".png"},
    "video": {".mp4", ".avi", ".mov"},
    "audio": {".mp3", ".wav", ".m4a", ".flac"},
    "document": {".pdf", ".docx", ".txt", ".xlsx"}
}

# Global models
photo_clip_model = None
photo_preprocess = None
doc_model = None

def load_models():
    global photo_clip_model, photo_preprocess, doc_model
    if photo_clip_model is None:
        photo_clip_model, photo_preprocess = clip.load("ViT-B/32", device=device, jit=False)
        photo_clip_model = photo_clip_model.half()
    if doc_model is None:
        doc_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Models loaded")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                tags TEXT,
                embedding BLOB,
                mime_type TEXT,
                hash TEXT UNIQUE
            )
        """)
        conn.commit()

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Copy model if not in /tmp
    if not os.path.exists(MODEL_PATH) and os.path.exists("/app/vosk-model-small-en-us-0.15"):
        shutil.copytree("/app/vosk-model-small-en-us-0.15", MODEL_PATH)
        logger.info(f"Copied model to {MODEL_PATH}")
    load_models()
    init_db()
    logger.info("Application started")
    yield
    # Shutdown (optional cleanup)
    logger.info("Application shutdown")

app.lifespan = lifespan

# Models
class IndexFilesRequest(BaseModel):
    file_paths: List[str]
    media_type: str

class SearchQuery(BaseModel):
    query: str
    media_type: Optional[str] = None
    top_k: Optional[int] = 10

class MediaResponse(BaseModel):
    id: int
    file_path: str
    file_type: str
    tags: Optional[str]
    similarity: float
    mime_type: str

# Utilities
def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_embedding(file_path: str, media_type: str, tags: Optional[str], embedding: np.ndarray, mime_type: str, file_hash: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM media WHERE hash = ?", (file_hash,))
        if cursor.fetchone():
            logger.info(f"Skipping duplicate: {file_path}")
            return
        cursor.execute(
            "INSERT INTO media (file_path, file_type, tags, embedding, mime_type, hash) VALUES (?, ?, ?, ?, ?, ?)",
            (file_path, media_type, tags, embedding.tobytes(), mime_type, file_hash)
        )
        conn.commit()

def get_all_files(root_path: str, extensions: set, max_files: int = 5000) -> List[str]:
    matched_files = []
    exclude_dirs = {"/system", "/data/data", "/android", "/proc", "/dev"}
    try:
        for dirpath, _, filenames in os.walk(root_path, followlinks=False):
            if any(dirpath.startswith(excl) for excl in exclude_dirs):
                continue
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in extensions:
                    full_path = os.path.join(dirpath, filename)
                    try:
                        if os.path.getsize(full_path) < 1024:
                            continue
                        matched_files.append(full_path)
                        if len(matched_files) >= max_files:
                            logger.warning(f"Reached max files limit ({max_files})")
                            return matched_files
                    except (OSError, PermissionError):
                        logger.warning(f"Skipping inaccessible file: {full_path}")
    except (PermissionError, OSError) as e:
        logger.error(f"Error scanning {root_path}: {e}")
    return matched_files

# Embedding functions
def embed_photo(file_path: str) -> np.ndarray:
    try:
        image = Image.open(file_path).convert("RGB")
        image_input = photo_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = photo_clip_model.encode_image(image_input).squeeze().cpu().numpy()
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        logger.error(f"Failed to embed photo {file_path}: {e}")
        return None

def embed_audio(file_path: str) -> np.ndarray:
    try:
        model = Model(MODEL_PATH)
        audio = AudioSegment.from_file(file_path).set_channels(1).set_frame_rate(16000)
        temp_wav = os.path.join(TRANSCRIPTIONS_FOLDER, f"{hashlib.md5(file_path.encode()).hexdigest()}.wav")
        audio.export(temp_wav, format="wav")
        wf = wave.open(temp_wav, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        result_text = ""
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                result_text += result.get("text", "") + " "
        final_result = json.loads(rec.FinalResult())
        result_text += final_result.get("text", "")
        os.remove(temp_wav)
        return doc_model.encode(result_text.strip()) if result_text.strip() else None
    except Exception as e:
        logger.error(f"Failed to embed audio {file_path}: {e}")
        return None

def embed_video(file_path: str) -> np.ndarray:
    try:
        cap = cv2.VideoCapture(file_path)
        embeddings = []
        frame_count = 0
        while len(embeddings) < 3:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 60 == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_input = photo_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = photo_clip_model.encode_image(image_input).squeeze().cpu().numpy()
                embeddings.append(embedding / np.linalg.norm(embedding))
            frame_count += 1
        cap.release()
        return np.mean(embeddings, axis=0) if embeddings else None
    except Exception as e:
        logger.error(f"Failed to embed video {file_path}: {e}")
        return None

def embed_document(file_path: str) -> np.ndarray:
    try:
        text = ""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
        elif ext == ".docx":
            doc = Document(file_path)
            text = " ".join(para.text for para in doc.paragraphs)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            text = " ".join(map(str, df.fillna("").values.flatten()))
        return doc_model.encode(text.strip()) if text.strip() else None
    except Exception as e:
        logger.error(f"Failed to embed document {file_path}: {e}")
        return None

# Endpoints
@app.post("/index-files")
async def index_files(req: IndexFilesRequest):
    if req.media_type not in SUPPORTED_MEDIA:
        raise HTTPException(status_code=400, detail="Invalid media type")

    embed_fn = {
        "photo": embed_photo,
        "audio": embed_audio,
        "video": embed_video,
        "document": embed_document
    }[req.media_type]
    indexed = []
    failed = []

    for i in range(0, len(req.file_paths), BATCH_SIZE):
        batch = req.file_paths[i:i + BATCH_SIZE]
        for path in batch:
            if not os.path.exists(path):
                failed.append({"file_path": path, "reason": "File not found"})
                continue
            if os.path.splitext(path)[1].lower() not in SUPPORTED_MEDIA[req.media_type]:
                failed.append({"file_path": path, "reason": "Unsupported format"})
                continue
            try:
                file_hash = compute_file_hash(path)
                dest_filename = f"{file_hash}{os.path.splitext(path)[1]}"
                dest_path = os.path.join(MEADOCS_FOLDER, dest_filename)
                shutil.copy2(path, dest_path)
                embedding = embed_fn(dest_path)
                if embedding is None:
                    failed.append({"file_path": path, "reason": "Embedding failed"})
                    os.remove(dest_path)
                    continue
                mime_type, _ = mimetypes.guess_type(dest_path)
                mime_type = mime_type or "application/octet-stream"
                save_embedding(dest_path, req.media_type, None, embedding, mime_type, file_hash)
                indexed.append(dest_path)
            except Exception as e:
                failed.append({"file_path": path, "reason": str(e)})
                logger.error(f"Failed to index {path}: {e}")

    return {"status": "success", "indexed": indexed, "failed": failed}

@app.post("/index-device")
async def index_entire_device(media_type: str):
    if media_type not in SUPPORTED_MEDIA:
        raise HTTPException(status_code=400, detail="Invalid media type")

    sample_dir = "/tmp/sample_media"  # Changed to /tmp
    file_paths = get_all_files(sample_dir, SUPPORTED_MEDIA[media_type])
    if not file_paths:
        raise HTTPException(status_code=404, detail="No media files found")

    logger.info(f"Found {len(file_paths)} {media_type} files")
    req = IndexFilesRequest(file_paths=file_paths, media_type=media_type)
    return await index_files(req)

@app.post("/search", response_model=List[MediaResponse])
async def search_media(query: SearchQuery):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    text_embedding = doc_model.encode(query.query.strip())
    results = []

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        sql = "SELECT id, file_path, file_type, tags, embedding, mime_type FROM media"
        params = []
        if query.media_type:
            sql += " WHERE file_type = ?"
            params.append(query.media_type)
        cursor.execute(sql, params)
        for row in cursor.fetchall():
            id_, file_path, file_type, tags, embedding_blob, mime_type = row
            if not os.path.exists(file_path):
                logger.warning(f"File missing: {file_path}")
                continue
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = np.dot(text_embedding, embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(embedding))
            results.append({
                "id": id_,
                "file_path": file_path,
                "file_type": file_type,
                "tags": tags,
                "similarity": float(similarity),
                "mime_type": mime_type
            })

    if not results:
        raise HTTPException(status_code=404, detail="No matching media found")

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:query.top_k]

@app.get("/serve/{media_id}")
async def serve_file(media_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, file_type, mime_type FROM media WHERE id = ?", (media_id,))
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="File not found")
        file_path, file_type, mime_type = result
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File no longer available")
        return {"file_path": file_path, "file_type": file_type, "mime_type": mime_type}

@app.get("/health")
async def health():
    return {"status": "Backend is running"}

@app.delete("/cleanup")
async def cleanup():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM media")
        for (file_path,) in cursor.fetchall():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
        cursor.execute("DELETE FROM media")
        conn.commit()
    return {"status": "Database and Meadocs_data cleared"}
