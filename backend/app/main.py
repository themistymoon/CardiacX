import io
import os
import sys
import tempfile
from pathlib import Path
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
# Import OpenAI with error handling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception as e:
    print(f"Error importing OpenAI: {e}")
    OpenAI = None
    OPENAI_AVAILABLE = False
from typing import Dict, List
import hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel

# --- App Setup ---
app = FastAPI(
    title="CardiacZ API",
    description="API for heart sound analysis and health chatbot.",
    version="1.0.0"
)

# --- Load Environment and Models ---
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. Chatbot functionality will be disabled.")

# Initialize OpenAI client only when needed to avoid proxy issues
client = None

# Clear any proxy-related environment variables that might interfere with OpenAI client
def clear_proxy_env():
    """Clear proxy environment variables that might interfere with OpenAI client."""
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']
    cleared_vars = []
    for var in proxy_vars:
        if var in os.environ:
            cleared_vars.append(var)
            del os.environ[var]
    if cleared_vars:
        print(f"Cleared proxy environment variables: {', '.join(cleared_vars)}")
    else:
        print("No proxy environment variables found to clear")
    
    # Also clear any other environment variables that might interfere
    # Check for any environment variables that might contain 'proxy' or 'PROXY'
    all_env_vars = list(os.environ.keys())
    for var in all_env_vars:
        if 'proxy' in var.lower() and var not in proxy_vars:
            print(f"Found additional proxy-related variable: {var}")
            del os.environ[var]
    
    # Clear any other environment variables that might interfere with OpenAI
    # Check for common variables that might be passed to HTTP clients
    http_vars = ['REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE', 'SSL_CERT_DIR']
    for var in http_vars:
        if var in os.environ:
            print(f"Clearing HTTP-related variable: {var}")
            del os.environ[var]

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR.parent / 'models'
MODEL_FILE = MODEL_DIR / 'my_model.h5'
LABELS_FILE = MODEL_DIR / 'labels.csv'

# Check if files exist
if not MODEL_FILE.exists():
    raise RuntimeError(f"Model file not found at {MODEL_FILE}")
if not LABELS_FILE.exists():
    raise RuntimeError(f"Labels file not found at {LABELS_FILE}")

# Initialize model and encoder as None for lazy loading
model = None
encoder = None
labels = None

def load_model_if_needed():
    """Load the ML model and encoder only when needed."""
    global model, encoder, labels
    if model is None:
        try:
            print("Loading ML model and encoder...")
            model = tf.keras.models.load_model(MODEL_FILE, custom_objects=None, compile=False)
            labels = pd.read_csv(LABELS_FILE)
            encoder = LabelEncoder()
            encoder.fit(labels['label'])
            print("ML model and encoder loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model or labels: {e}")

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

# --- Core Logic (from old app) ---

class HeartHealthChatbot:
    def __init__(self):
        self.conversation_history = []
        
    def generate_response(self, user_message: str, context: Dict = None, history: List[ChatHistory] = []) -> str:
        """Generate AI response for heart health queries."""
        global client
        if not OPENAI_API_KEY:
            return "Chatbot is disabled because the OpenAI API key is missing."
        if not OPENAI_AVAILABLE:
            return "Chatbot is disabled because the OpenAI library is not available."
        try:
            # Create client only when needed - simplified approach
            if client is None and OPENAI_AVAILABLE:
                try:
                    # Clear proxy environment variables that might interfere
                    clear_proxy_env()
                    # Debug: Print the OpenAI class signature
                    import inspect
                    sig = inspect.signature(OpenAI.__init__)
                    print(f"OpenAI.__init__ signature: {sig}")
                    
                    # Try to create client with explicit parameter passing
                    # Use **kwargs to avoid any unexpected parameters
                    client_kwargs = {'api_key': OPENAI_API_KEY}
                    print(f"Creating OpenAI client with kwargs: {list(client_kwargs.keys())}")
                    
                    # Check if there are any global configurations that might interfere
                    import openai
                    if hasattr(openai, '_client'):
                        print("Found global OpenAI client configuration")
                    if hasattr(openai, 'api_key'):
                        print("Found global OpenAI API key")
                    
                    # Try creating the client in a completely isolated way
                    # Save current environment
                    original_env = dict(os.environ)
                    
                    # Clear all potentially problematic environment variables
                    clear_proxy_env()
                    
                    # Try to create client with minimal environment
                    try:
                        # Try to create client without any parameters first
                        client = OpenAI()
                        client.api_key = OPENAI_API_KEY
                    except Exception as inner_e:
                        print(f"Inner error: {inner_e}")
                        # Try a different approach - use the raw constructor
                        try:
                            # Import the client class directly
                            from openai._client import OpenAI as RawOpenAI
                            client = RawOpenAI(api_key=OPENAI_API_KEY)
                        except Exception as raw_e:
                            print(f"Raw client creation failed: {raw_e}")
                            # Try using httpx directly to bypass the client
                            try:
                                import httpx
                                # Create a simple HTTP client for OpenAI API
                                http_client = httpx.Client(
                                    base_url="https://api.openai.com/v1",
                                    headers={
                                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                                        "Content-Type": "application/json"
                                    }
                                )
                                # Create a simple wrapper
                                class SimpleOpenAIClient:
                                    def __init__(self, http_client):
                                        self.http_client = http_client
                                    
                                    def chat(self):
                                        return SimpleChatCompletions(self.http_client)
                                
                                class SimpleChatCompletions:
                                    def __init__(self, http_client):
                                        self.http_client = http_client
                                    
                                    def create(self, **kwargs):
                                        response = self.http_client.post("/chat/completions", json=kwargs)
                                        response.raise_for_status()
                                        return SimpleResponse(response.json())
                                
                                class SimpleResponse:
                                    def __init__(self, data):
                                        self.data = data
                                    
                                    @property
                                    def choices(self):
                                        return [SimpleChoice(choice) for choice in self.data.get("choices", [])]
                                
                                class SimpleChoice:
                                    def __init__(self, choice_data):
                                        self.choice_data = choice_data
                                    
                                    @property
                                    def message(self):
                                        return SimpleMessage(self.choice_data.get("message", {}))
                                
                                class SimpleMessage:
                                    def __init__(self, message_data):
                                        self.message_data = message_data
                                    
                                    @property
                                    def content(self):
                                        return self.message_data.get("content", "")
                                
                                client = SimpleOpenAIClient(http_client)
                            except Exception as http_e:
                                print(f"HTTP client creation failed: {http_e}")
                                # Last resort - try with minimal kwargs
                                client = OpenAI(api_key=OPENAI_API_KEY)
                except Exception as e:
                    print(f"Error creating OpenAI client: {e}")
                    # Try alternative initialization without any extra parameters
                    try:
                        clear_proxy_env()
                        client = OpenAI()
                        client.api_key = OPENAI_API_KEY
                    except Exception as e2:
                        print(f"Alternative OpenAI client creation also failed: {e2}")
                        # Try a third approach - create client with minimal configuration
                        try:
                            clear_proxy_env()
                            # Try creating client without any parameters first
                            client = OpenAI()
                            client.api_key = OPENAI_API_KEY
                        except Exception as e3:
                            print(f"Third OpenAI client creation attempt also failed: {e3}")
                            raise e
            elif client is None:
                return "Chatbot is temporarily unavailable due to configuration issues. Please try again later or contact support."
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
                {"role": "system", "content": system_prompt + context_info}
            ]
            for msg in history:
                # Replace 'bot' with 'assistant' for API compatibility
                role = "assistant" if msg.role == "bot" else msg.role
                messages.append({"role": role, "content": msg.text})
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # In a real app, you'd want to log this error more robustly
            print(f"OpenAI API error: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            return "I'm sorry, I'm having trouble connecting to my knowledge base right now. For any urgent health concerns, please consult a medical professional."

chatbot = HeartHealthChatbot()

def extract_heart_sound(audio):
    fourier_transform = np.fft.fft(audio)
    heart_sound = np.abs(fourier_transform)
    return heart_sound

def preprocess_audio(file_bytes: bytes, file_format: str):
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
        audio = y / np.max(np.abs(y))
        # Extract heart sound using Fourier transform (as in original code, though its output is not used)
        heart_sound = extract_heart_sound(audio)
        # Generate the spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram)
        # Define a fixed length for the spectrogram
        fixed_length = 1000  # Adjust this value as necessary
        # Pad or truncate the spectrogram to the fixed length
        if (spectrogram.shape[1] > fixed_length):
            spectrogram = spectrogram[:, :fixed_length]
        else:
            padding = fixed_length - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), 'constant')
        # Reshape the spectrogram to fit the model
        spectrogram = spectrogram.reshape((1, 128, 1000, 1))
        return spectrogram
    finally:
        # Clean up the temporary files
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
    print("Note: ML model will be loaded on first request for better startup performance")
    # Clear proxy environment variables on startup
    clear_proxy_env()
    
    # Check OpenAI availability
    if OPENAI_AVAILABLE:
        print("OpenAI library is available - chatbot functionality enabled")
    else:
        print("OpenAI library is not available - chatbot functionality disabled")
    
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set - chatbot functionality will be disabled")

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyzes an uploaded audio file of a heart sound.
    
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
    
    # mpeg is a common content type for mp3
    if file_format == 'mpeg':
        file_format = 'mp3'

    if file_format not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"File format '{file_format}' not supported.")

    try:
        # Load model if not already loaded
        load_model_if_needed()
        
        spectrogram = preprocess_audio(file_bytes, file_format)
        
        if spectrogram is None:
            raise HTTPException(status_code=500, detail="Error processing audio file.")

        # Get prediction probabilities
        y_pred = model.predict(spectrogram)
        class_probabilities = y_pred[0].tolist() # Convert to list for JSON serialization
        
        # Create a dictionary of class probabilities
        prob_dict = {label: prob for label, prob in zip(encoder.classes_, class_probabilities)}

        # Determine the predicted label
        sorted_indices = np.argsort(-y_pred[0])
        predicted_label = encoder.inverse_transform([sorted_indices[0]])[0]
        confidence_score = y_pred[0][sorted_indices[0]]

        # Handle artifact case
        primary_prediction = predicted_label
        primary_confidence = confidence_score
        is_artifact = False
        if predicted_label == 'artifact' and confidence_score >= 0.70:
            is_artifact = True
        elif predicted_label == 'artifact':
            # If artifact is top but confidence is low, take the next one
            primary_prediction = encoder.inverse_transform([sorted_indices[1]])[0]
            primary_confidence = y_pred[0][sorted_indices[1]]

        # Get condition information
        condition_info = HEART_CONDITIONS.get(primary_prediction, {
            "description": "Condition information not available.",
            "severity": "Unknown",
            "recommendations": [],
            "urgency": "Consult a professional"
        })
        
        response_data = {
            "predicted_condition": primary_prediction,
            "confidence": round(float(primary_confidence) * 100, 2),
            "medical_info": condition_info,
            "is_artifact": is_artifact,
            "probabilities": prob_dict
        }
        
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
    """Handles chat messages to the AI assistant."""
    try:
        response = chatbot.generate_response(chat_message.message, history=chat_message.history)
        return {"response": response}
    except Exception as e:
        print(f"Error in assistant chat: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the assistant.")

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 