"""
AI Voice Detection API - OPTIMIZED for Render deployment
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
import time

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


# Feature Extraction Class - OPTIMIZED
class AudioFeatureExtractor:
    """Extract advanced features for AI voice detection - OPTIMIZED"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    def extract_mfcc_features(self, audio):
        """Extract MFCC features - optimized"""
        try:
            # Reduce n_fft for speed
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=20, n_fft=1024, hop_length=512)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            return np.concatenate([mfccs_mean, mfccs_std]).astype(np.float32)
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return np.zeros(40, dtype=np.float32)
    
    def extract_spectral_features(self, audio):
        """Extract spectral features - optimized"""
        try:
            # Use same hop_length for speed
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr, n_fft=1024, hop_length=512)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr, n_fft=1024, hop_length=512)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr, n_fft=1024, hop_length=512)
            
            features = np.array([
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
            ], dtype=np.float32)
            
            contrast_mean = np.mean(spectral_contrast, axis=1).astype(np.float32)
            contrast_std = np.std(spectral_contrast, axis=1).astype(np.float32)
            
            return np.concatenate([features, contrast_mean, contrast_std])
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {e}")
            return np.zeros(18, dtype=np.float32)
    
    def extract_pitch_features(self, audio):
        """Extract pitch-related features - simplified for speed"""
        try:
            # Simplified pitch tracking
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr, n_fft=1024, hop_length=512)
            
            pitch_values = []
            for t in range(min(pitches.shape[1], 100)):  # Limit frames for speed
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) == 0:
                return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            pitch_values = np.array(pitch_values)
            return np.array([
                np.mean(pitch_values),
                np.std(pitch_values),
                np.max(pitch_values),
                np.min(pitch_values)
            ], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def extract_zero_crossing_rate(self, audio):
        """Zero crossing rate - fast"""
        try:
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=512)
            return np.array([np.mean(zcr), np.std(zcr)], dtype=np.float32)
        except Exception as e:
            logger.warning(f"ZCR extraction failed: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def extract_energy_features(self, audio):
        """Energy features - fast"""
        try:
            rms = librosa.feature.rms(y=audio, hop_length=512)
            return np.array([np.mean(rms), np.std(rms)], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Energy extraction failed: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def extract_phase_features(self, audio):
        """Phase consistency - simplified"""
        try:
            stft = librosa.stft(audio, n_fft=1024, hop_length=512)
            phase = np.angle(stft)
            phase_diff = np.diff(phase, axis=1)
            
            return np.array([
                np.mean(np.abs(phase_diff)),
                np.std(np.abs(phase_diff)),
                np.max(np.abs(phase_diff))
            ], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Phase extraction failed: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def extract_all_features(self, audio):
        """Extract all features and combine - OPTIMIZED"""
        # Downsample audio if too long for speed
        max_length = self.sr * 10  # Max 10 seconds
        if len(audio) > max_length:
            audio = audio[:max_length]
        
        features = []
        
        features.append(self.extract_mfcc_features(audio))
        features.append(self.extract_spectral_features(audio))
        features.append(self.extract_pitch_features(audio))
        features.append(self.extract_zero_crossing_rate(audio))
        features.append(self.extract_energy_features(audio))
        features.append(self.extract_phase_features(audio))
        
        all_features = np.concatenate(features)
        
        # Pad or trim to exactly 69 features (adjusted for faster extraction)
        target_size = 69
        if len(all_features) < target_size:
            all_features = np.pad(all_features, (0, target_size - len(all_features)))
        elif len(all_features) > target_size:
            all_features = all_features[:target_size]
        
        return all_features.astype(np.float32)


# Neural Network Model - ADJUSTED for new feature size
class VoiceClassifier(nn.Module):
    """Neural network for voice classification"""
    
    def __init__(self, input_size=69):
        super(VoiceClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
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
    """Decode base64 audio to numpy array - OPTIMIZED"""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load with faster settings
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=16000, 
            mono=True,
            duration=30  # Limit to 30 seconds max
        )
        
        return audio
    except Exception as e:
        logger.error(f"Error decoding audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")


def predict_voice_type(audio: np.ndarray, language: str) -> tuple:
    """
    Predict if voice is AI-generated or human - OPTIMIZED
    Returns: (classification, confidence, explanation)
    """
    try:
        # Extract features
        features = feature_extractor.extract_all_features(audio)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        if model is not None:
            # Use trained model
            with torch.no_grad():
                output = model(features_tensor)
                ai_probability = float(output.item())
                
                classification = "AI_GENERATED" if ai_probability > 0.5 else "HUMAN"
                confidence = abs(ai_probability - 0.5) * 2
                
                explanation = {
                    "ai_probability": ai_probability,
                    "model": "trained_neural_network"
                }
        else:
            # Improved heuristic fallback
            logger.warning("Using heuristic model")
            
            # Better heuristics based on features
            mfcc_std = float(np.std(features[:20]))
            spectral_std = float(features[21]) if len(features) > 21 else 0.0
            pitch_std = float(features[42]) if len(features) > 42 else 50.0
            
            ai_score = 0.5
            
            # Low MFCC variation suggests AI
            if mfcc_std < 5.0:
                ai_score += 0.15
            
            # Low spectral variation suggests AI
            if spectral_std < 500:
                ai_score += 0.15
            
            # Abnormal pitch suggests AI
            if pitch_std < 30 or pitch_std > 200:
                ai_score += 0.15
            
            ai_score = np.clip(ai_score, 0.0, 1.0)
            confidence = abs(ai_score - 0.5) * 2
            confidence = np.clip(confidence, 0.65, 0.95)
            
            classification = "AI_GENERATED" if ai_score > 0.5 else "HUMAN"
            
            explanation = {
                "mfcc_variation": mfcc_std,
                "spectral_variation": spectral_std,
                "pitch_variation": pitch_std,
                "ai_score": float(ai_score),
                "model": "heuristic_baseline"
            }
        
        return classification, float(confidence), explanation
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global model, feature_extractor
    
    logger.info(f"Starting server on device: {device}")
    
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor()
    logger.info("✓ Feature extractor initialized")
    
    # Try to load model
    model_path = "voice_classifier_best.pth"
    
    if os.path.exists(model_path):
        try:
            logger.info(f"Loading model from {model_path}...")
            model = VoiceClassifier(input_size=69).to(device)
            
            # Load with weights_only=True for security
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info("✓ Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.warning("Will use heuristic-based prediction")
            model = None
    else:
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("Will use heuristic-based prediction")
        model = None
    
    yield
    
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated vs human voices in 5 languages",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "supported_languages": SUPPORTED_LANGUAGES,
        "model_loaded": model is not None
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
    """
    start_time = time.time()
    
    # Validate API key FIRST
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
        
        # Validate audio length
        duration = len(audio) / 16000
        if duration < 0.5 or duration > 60:
            raise HTTPException(
                status_code=400, 
                detail=f"Audio duration must be between 0.5 and 60 seconds (got {duration:.2f}s)"
            )
        
        # Predict
        classification, confidence, explanation = predict_voice_type(audio, request.language)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Processed request in {processing_time:.2f}ms - {classification} ({confidence:.4f})")
        
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
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
