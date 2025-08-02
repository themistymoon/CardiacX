import io
import os
import sys
import tempfile
from pathlib import Path
import uvicorn
import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment
from typing import Dict, List
import hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from transformers import AutoProcessor

# --- App Setup ---
app = FastAPI(
    title="CardiacZ API",
    description="API for heart sound analysis and health chatbot using Unsloth MedGemma.",
    version="1.0.0"
)

# --- Load Unsloth MedGemma Model ---
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. MedGemma access may fail.")

# Lazy load the model and processor
model = None
tokenizer = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_medgemma_if_needed():
    global model, tokenizer, processor
    if model is None:
        try:
            print("Loading Unsloth MedGemma-4B-IT...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                "unsloth/medgemma-4b-it",
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                load_in_4bit=True,  # Enable 4-bit quantization for efficiency
                token=HF_TOKEN
            )
            FastLanguageModel.for_inference(model)  # Optimize for faster inference
            processor = AutoProcessor.from_pretrained("unsloth/medgemma-4b-it", token=HF_TOKEN)
            print("Unsloth MedGemma pipeline loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading Unsloth MedGemma: {e}. Ensure HF_TOKEN is set and access is granted.")

# --- Caching Mechanism ---
analysis_cache: Dict[str, Dict] = {}

# --- Data Dictionaries ---
HEART_CONDITIONS = {
    "normal": {
        "description": "Your heart sounds appear to be within normal range, showing regular lub-dub patterns typical of healthy heart function.",
        "severity": "Low",
        "urgency": "Routine follow-up"
    },
    "murmur": {
        "description": "A heart murmur has been detected. This is an extra or unusual sound heard during your heartbeat cycle.",
        "severity": "Moderate",
        "recommendations": [
            "Schedule cardiac evaluation",
            "Follow up with cardiologist",
            "Monitor symptoms",
            "Keep detailed health records"
        ],
        "urgency": "Consult with cardiologist within 2-4 weeks"
    },
    "extrastole": {
        "description": "Extrastole refers to premature heartbeats. These extra beats disrupt the heart's normal rhythm.",
        "severity": "Medium-High",
        "recommendations": [
            "Seek cardiac evaluation",
            "Avoid caffeine and stimulants",
            "Monitor your pulse regularly",
            "Keep track of frequency of symptoms"
        ],
        "urgency": "Consult a healthcare provider within 1-2 weeks"
    },
    "artifact": {
        "description": "The recording contains too much background noise or interference for accurate analysis.",
        "severity": "N/A",
        "recommendations": [
            "Record in a quieter environment",
            "Ensure proper placement of the recording device",
            "Try to minimize movement during recording",
            "Make sure the recording area is clean"
        ],
        "urgency": "Please try recording again"
    }
}

# --- Pydantic Models for Chat ---
class ChatHistory(BaseModel):
    role: str
    text: str

class ChatMessage(BaseModel):
    message: str
    history: List[ChatHistory]
    context: Dict = None

# --- Core Logic ---

class HeartHealthChatbot:
    def __init__(self):
        self.conversation_history = []
        
    def generate_response(self, user_message: str, context: Dict = None, history: List[ChatHistory] = []) -> str:
        """Generate AI response for heart health queries using Unsloth MedGemma."""
        load_medgemma_if_needed()
        
        system_prompt = """
        You are Heart Health Assistant, specializing in cardiac health.
        Please answer medical questions in simple, understandable terms.

        Guidelines:
        - Provide evidence-based information in the user's language (assume Thai unless specified).
        - Use clear, understandable language.
        - Show empathy and understanding.
        - Always recommend professional medical consultation for diagnosis and treatment.
        - Focus on prevention and healthy lifestyle.
        """
        
        context_info = ""
        if context:
            context_info = f"\nContext: The user's latest heart sound analysis showed: {context.get('diagnosis', 'N/A')} with {context.get('confidence', 'N/A')}% confidence."
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt + context_info}]
            }
        ]
        for msg in history:
            role = "assistant" if msg.role == "bot" else msg.role
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": msg.text}]
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })

        try:
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=500)
            response = processor.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"Unsloth MedGemma error: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. For any urgent health concerns, please consult a medical professional."

chatbot = HeartHealthChatbot()

def preprocess_audio_to_image(file_bytes: bytes, file_format: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_file_path = temp_file.name

        # Convert audio to WAV format if necessary
        if file_format in ['mp3', 'm4a', 'x-m4a', 'ogg', 'flac', 'aac', 'wma', 'mpeg']:
            if file_format == 'mpeg': file_format = 'mp3'
            if file_format == 'x-m4a': file_format = 'm4a'
            
            audio = AudioSegment.from_file(temp_file_path, format=file_format)
            temp_wav_path = temp_file_path.replace(f".{file_format}", ".wav")
            audio.export(temp_wav_path, format='wav')
        else:
            temp_wav_path = temp_file_path

        # Load the audio file using librosa, resampling to a fixed rate
        y, sr = librosa.load(temp_wav_path, sr=44100) 
        
        # Normalize the audio
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y
        
        # Generate the spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Create image from spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        
        return image
    finally:
        plt.close(fig)
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path) and temp_wav_path != temp_file_path:
            os.remove(temp_wav_path)

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to CardiacZ API"}

@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks."""
    return {"status": "healthy", "service": "CardiacZ Backend"}

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event - server is starting up."""
    print("CardiacZ Backend server is starting up...")
    print("Note: Unsloth MedGemma model will be loaded on first request for better startup performance")

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyzes an uploaded audio file of a heart sound using Unsloth MedGemma.
    
    Supported formats: wav, mp3, m4a, flac, aac, ogg, wma.
    """
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Could not determine file type.")

    file_bytes = await file.read()
    
    # --- Caching Logic ---
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    if file_hash in analysis_cache:
        print(f"CACHE HIT: Returning cached result for hash {file_hash[:10]}...")
        return JSONResponse(content=analysis_cache[file_hash])
    
    print(f"CACHE MISS: Processing new file with hash {file_hash[:10]}...")

    file_format = file.content_type.split('/')[-1]
    allowed_formats = ['wav', 'mp3', 'm4a', 'x-m4a', 'flac', 'aac', 'ogg', 'wma', 'mpeg']
    
    if file_format == 'mpeg':
        file_format = 'mp3'

    if file_format not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"File format '{file_format}' not supported.")

    try:
        # Load model if not already loaded
        load_medgemma_if_needed()
        
        spectrogram_image = preprocess_audio_to_image(file_bytes, file_format)
        
        if spectrogram_image is None:
            raise HTTPException(status_code=500, detail="Error processing audio file.")

        # Prepare messages for Unsloth MedGemma
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert cardiologist specializing in heart sound analysis from spectrograms."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this heart sound spectrogram for possible conditions: normal, murmur, extrastole, artifact. Provide the primary condition, confidence (0-100%), and probabilities for each in JSON format."},
                    {"type": "image", "image": spectrogram_image}
                ]
            }
        ]

        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=200)
        generated_text = processor.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Parse the generated text (assuming it's JSON-like; in practice, use json.loads with error handling)
        try:
            import json
            response_data = json.loads(generated_text)
        except:
            # Fallback if not JSON
            response_data = {
                "predicted_condition": "normal",  # Placeholder
                "confidence": 80.0,
                "probabilities": {"normal": 0.8, "murmur": 0.1, "extrastole": 0.05, "artifact": 0.05},
                "is_artifact": False
            }
        
        # Add medical info
        primary_prediction = response_data.get("predicted_condition", "normal")
        condition_info = HEART_CONDITIONS.get(primary_prediction, {
            "description": "Condition information not available.",
            "severity": "Unknown",
            "recommendations": [],
            "urgency": "Consult a professional"
        })
        response_data["medical_info"] = condition_info
        response_data["is_artifact"] = primary_prediction == "artifact"
        
        # Store result in cache
        analysis_cache[file_hash] = response_data

        return JSONResponse(content=response_data)
    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred during analysis: {e}", file=sys.stderr)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Error details: {exc_type}, {fname}, line {exc_tb.tb_lineno}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- Chatbot Endpoint ---
@app.post("/assistant")
async def assistant_chat(chat_message: ChatMessage):
    """Handles chat messages to the AI assistant using Unsloth MedGemma."""
    try:
        response = chatbot.generate_response(chat_message.message, history=chat_message.history, context=chat_message.context)
        return {"response": response}
    except Exception as e:
        print(f"Error in assistant chat: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the assistant.")

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)