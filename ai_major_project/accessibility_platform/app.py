import io
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2
from typing import Optional
import asyncio

# Import model wrappers
from models.sign_language import SignLanguageRecognizer
from models.scene_desc import SceneDescriber
from models.speech_to_text import SpeechToTextImpaired
from models.text_simplify import TextSimplifier
from utils.whatsapp_integration import send_whatsapp_message
from utils.gmail_integration import send_email

app = FastAPI(title="Accessibility Platform for Differently-Abled")

# Initialize models (lazy loading for memory efficiency)
sign_model = None
scene_model = None
stt_model = None
simplify_model = None

def get_sign_model():
    global sign_model
    if sign_model is None:
        sign_model = SignLanguageRecognizer()
    return sign_model

def get_scene_model():
    global scene_model
    if scene_model is None:
        scene_model = SceneDescriber()
    return scene_model

def get_stt_model():
    global stt_model
    if stt_model is None:
        stt_model = SpeechToTextImpaired()
    return stt_model

def get_simplify_model():
    global simplify_model
    if simplify_model is None:
        simplify_model = TextSimplifier()
    return simplify_model

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------- 1. Sign Language Recognition ----------
@app.post("/predict_sign")
async def predict_sign(file: UploadFile = File(...)):
    """Receives an image (hand gesture) and returns recognised ISL word/phrase."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    # Convert PIL to numpy (OpenCV format)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    model = get_sign_model()
    text_result = model.predict(frame)
    # Also generate speech audio (return as base64 wav)
    audio_b64 = model.text_to_speech(text_result)
    
    return {"text": text_result, "audio_base64": audio_b64}

# ---------- 2. Scene Description ----------
@app.post("/describe_scene")
async def describe_scene(file: UploadFile = File(...)):
    """Describe the current camera frame for visually impaired users."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    model = get_scene_model()
    description = model.describe(image)
    # Generate audio description
    audio_b64 = model.text_to_speech(description)
    return {"description": description, "audio_base64": audio_b64}

# ---------- 3. Speech-to-Text for Impaired Speech ----------
@app.post("/stt_impaired")
async def stt_impaired(file: UploadFile = File(...)):
    """Transcribe audio (WAV/MP3) containing impaired speech."""
    contents = await file.read()
    # Save temporarily or process in memory
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp.flush()
        model = get_stt_model()
        text = model.transcribe(tmp.name)
    return {"text": text}

# ---------- 4. Text Simplification ----------
@app.post("/simplify_text")
async def simplify_text(text: str = Form(...)):
    """Simplify complex English text for learning disabilities."""
    model = get_simplify_model()
    simplified = model.simplify(text)
    return {"original": text, "simplified": simplified}

# ---------- Integrations ----------
@app.post("/send_whatsapp")
async def send_whatsapp(phone_number: str = Form(...), message: str = Form(...)):
    try:
        send_whatsapp_message(phone_number, message)
        return {"status": "sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_email")
async def send_email_endpoint(to: str = Form(...), subject: str = Form(...), body: str = Form(...)):
    try:
        send_email(to, subject, body)
        return {"status": "sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)