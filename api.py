"""
AI Voice Detection API - Competition Winning Implementation
Supports: Tamil, English, Hindi, Malayalam, Telugu
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
import io
import numpy as np
import librosa
import torch
import torch.nn as nn
from typing import Literal, Optional
import json
from datetime import datetime
from contextlib import asynccontextmanager
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
VALID_API_KEY = "gnGFZWm0qX7w-K__uZnOg7XJfwS3ZbnFTUGJqax-cKw"
SUPPORTED_LANGUAGES = ["tamil", "english", "hindi", "malayalam", "telugu"]

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = None

# Request/Response Models
class VoiceDetectionRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio")
    language: Literal["tamil", "english", "hindi", "malayalam", "telugu"] = Field(
        default="english", 
        description="Language of the audio sample"
    )

class VoiceDetectionResponse(BaseModel):
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    language: str
    processing_time_ms: float
    model_version: str = "1.0.0"
    explanation: Optional[dict] = None


# Feature Extraction Class
class AudioFeatureExtractor:
    """Extract advanced features for AI voice detection"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    def extract_mfcc_features(self, audio):
        """Extract MFCC features"""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            return np.concatenate([mfccs_mean, mfccs_std]).astype(float)
        except:
            return np.zeros(80, dtype=float)
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
            
            features = np.array([
                float(np.mean(spectral_centroid)),
                float(np.std(spectral_centroid)),
                float(np.mean(spectral_rolloff)),
                float(np.std(spectral_rolloff)),
            ])
            
            contrast_mean = np.mean(spectral_contrast, axis=1).astype(float)
            contrast_std = np.std(spectral_contrast, axis=1).astype(float)
            
            return np.concatenate([features, contrast_mean, contrast_std])
        except:
            return np.zeros(18, dtype=float)
    
    def extract_pitch_features(self, audio):
        """Extract pitch-related features (AI often has unnatural pitch)"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr)
            
            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) == 0:
                return np.array([0.0, 0.0, 0.0, 0.0])
            
            pitch_values = np.array(pitch_values)
            return np.array([
                float(np.mean(pitch_values)),
                float(np.std(pitch_values)),
                float(np.max(pitch_values)),
                float(np.min(pitch_values))
            ])
        except:
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def extract_zero_crossing_rate(self, audio):
        """Zero crossing rate - AI voices often have different patterns"""
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            return np.array([float(np.mean(zcr)), float(np.std(zcr))])
        except:
            return np.array([0.0, 0.0])
    
    def extract_energy_features(self, audio):
        """Energy features - AI voices may have unnatural energy distribution"""
        try:
            rms = librosa.feature.rms(y=audio)
            return np.array([float(np.mean(rms)), float(np.std(rms))])
        except:
            return np.array([0.0, 0.0])
    
    def extract_phase_features(self, audio):
        """Phase consistency - AI voices often have phase artifacts"""
        try:
            stft = librosa.stft(audio)
            phase = np.angle(stft)
            
            # Phase derivative (discontinuities indicate AI)
            phase_diff = np.diff(phase, axis=1)
            
            return np.array([
                float(np.mean(np.abs(phase_diff))),
                float(np.std(np.abs(phase_diff))),
                float(np.max(np.abs(phase_diff)))
            ])
        except:
            return np.array([0.0, 0.0, 0.0])
    
    def extract_all_features(self, audio):
        """Extract all features and combine"""
        features = []
        
        features.append(self.extract_mfcc_features(audio))
        features.append(self.extract_spectral_features(audio))
        features.append(self.extract_pitch_features(audio))
        features.append(self.extract_zero_crossing_rate(audio))
        features.append(self.extract_energy_features(audio))
        features.append(self.extract_phase_features(audio))
        
        # Concatenate all features
        all_features = np.concatenate(features)
        
        # Ensure we have exactly 109 features (to match trained model)
        if len(all_features) < 109:
            all_features = np.pad(all_features, (0, 109 - len(all_features)))
        elif len(all_features) > 109:
            all_features = all_features[:109]
        
        return all_features.astype(float)


# Neural Network Model (matching the trained model architecture)
class VoiceClassifier(nn.Module):
    """Neural network for voice classification"""
    
    def __init__(self, input_size=109):
        super(VoiceClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


def decode_audio(audio_base64: str) -> np.ndarray:
    """Decode base64 audio to numpy array"""
    try:
        # Decode base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio with librosa
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        
        return audio
    except Exception as e:
        logger.error(f"Error decoding audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")


def predict_voice_type(audio: np.ndarray, language: str) -> tuple:
    """
    Predict if voice is AI-generated or human
    Returns: (classification, confidence, explanation)
    """
    try:
        # Extract features
        features = feature_extractor.extract_all_features(audio)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # Check if model is loaded
        if model is not None:
            # Use trained model
            with torch.no_grad():
                output = model(features_tensor)
                ai_probability = output.item()
                
                classification = "AI_GENERATED" if ai_probability > 0.5 else "HUMAN"
                confidence = abs(ai_probability - 0.5) * 2  # Scale to 0-1
                
                explanation = {
                    "ai_probability": float(ai_probability),
                    "model": "trained_neural_network"
                }
        else:
            # Use heuristic-based prediction (fallback)
            logger.warning("Using heuristic model - trained model not loaded")
            
            # Get feature statistics
            pitch_std = features[82] if len(features) > 82 else 50.0
            phase_mean = features[103] if len(features) > 103 else 1.0
            energy_std = features[101] if len(features) > 101 else 0.02
            
            # Simple heuristic scoring
            ai_score = 0.5
            
            # Low pitch variation suggests AI
            if pitch_std < 50:
                ai_score += 0.2
            
            # High phase discontinuity suggests AI
            if phase_mean > 1.5:
                ai_score += 0.2
            
            # Low energy variation suggests AI
            if energy_std < 0.015:
                ai_score += 0.1
            
            # Ensure score is in valid range
            ai_score = max(0.0, min(1.0, ai_score))
            
            confidence = abs(ai_score - 0.5) * 2
            confidence = max(0.6, min(0.95, confidence))
            
            classification = "AI_GENERATED" if ai_score > 0.5 else "HUMAN"
            
            explanation = {
                "pitch_variability": float(pitch_std),
                "phase_consistency": float(phase_mean),
                "energy_variation": float(energy_std),
                "ai_score": float(ai_score),
                "model": "heuristic_baseline"
            }
        
        return classification, confidence, explanation
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global model, feature_extractor
    
    # Startup
    logger.info(f"Starting server on device: {device}")
    
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor()
    
    # Try to load model
    model_path = "voice_classifier_best.pth"
    
    if os.path.exists(model_path):
        try:
            logger.info(f"Loading model from {model_path}...")
            model = VoiceClassifier(input_size=109).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            logger.info("✓ Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.warning("Will use heuristic-based prediction instead")
            model = None
    else:
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("Will use heuristic-based prediction instead")
        model = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated vs human voices in 5 languages",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "supported_languages": SUPPORTED_LANGUAGES
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "device": str(device),
        "supported_languages": SUPPORTED_LANGUAGES
    }


@app.post("/detect", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(..., description="API Key for authentication")
):
    """
    Main endpoint for voice detection
    
    Args:
        request: VoiceDetectionRequest with base64 audio and language
        x_api_key: API key for authentication
    
    Returns:
        VoiceDetectionResponse with classification and confidence
    """
    start_time = datetime.utcnow()
    
    # Validate API key
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language. Supported: {SUPPORTED_LANGUAGES}"
        )
    
    try:
        # Decode audio
        audio = decode_audio(request.audio_base64)
        
        # Validate audio length (should be reasonable)
        duration = len(audio) / 16000  # seconds
        if duration < 0.5 or duration > 60:
            raise HTTPException(
                status_code=400, 
                detail=f"Audio duration must be between 0.5 and 60 seconds (got {duration:.2f}s)"
            )
        
        # Predict
        classification, confidence, explanation = predict_voice_type(audio, request.language)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return VoiceDetectionResponse(
            classification=classification,
            confidence=round(confidence, 4),
            language=request.language,
            processing_time_ms=round(processing_time, 2),
            explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
