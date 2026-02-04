"""
Model Training Script - Advanced AI Voice Detection
This script trains multiple models and creates an ensemble
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AudioDataset(Dataset):
    """Custom dataset for audio samples"""
    
    def __init__(self, audio_paths, labels, feature_extractor, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.augment = augment
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_paths[idx]
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Apply data augmentation if enabled
        if self.augment:
            audio = self.augment_audio(audio)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio)
        
        label = self.labels[idx]
        
        return torch.FloatTensor(features), torch.FloatTensor([label])
    
    def augment_audio(self, audio):
        """Apply data augmentation"""
        augmentations = [
            lambda x: self.time_stretch(x),
            lambda x: self.pitch_shift(x),
            lambda x: self.add_noise(x),
        ]
        
        # Apply random augmentation
        if np.random.random() > 0.5:
            aug_func = np.random.choice(augmentations)
            audio = aug_func(audio)
        
        return audio
    
    def time_stretch(self, audio, rate_range=(0.9, 1.1)):
        """Time stretching"""
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps_range=(-2, 2)):
        """Pitch shifting"""
        n_steps = np.random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(audio, sr=16000, n_steps=n_steps)
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add random noise"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise


class ImprovedVoiceClassifier(nn.Module):
    """Enhanced neural network with better architecture"""
    
    def __init__(self, input_size=107, dropout=0.3):
        super(ImprovedVoiceClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # First block
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third block
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            
            # Fourth block
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Output
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class CNNSpectrogramClassifier(nn.Module):
    """CNN-based classifier using mel spectrograms"""
    
    def __init__(self):
        super(CNNSpectrogramClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 8, 256),  # Adjust based on input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """Train the model"""
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_probs.extend(outputs.cpu().numpy())
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model! Val Acc: {best_val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


def collect_dataset_paths(data_dir):
    """
    Collect paths to audio files and their labels
    
    Expected directory structure:
    data_dir/
        human/
            tamil/
            english/
            hindi/
            malayalam/
            telugu/
        ai_generated/
            tamil/
            english/
            hindi/
            malayalam/
            telugu/
    """
    audio_paths = []
    labels = []
    
    # Collect human voices (label = 0)
    human_dir = os.path.join(data_dir, "human")
    if os.path.exists(human_dir):
        for lang in ["tamil", "english", "hindi", "malayalam", "telugu"]:
            lang_dir = os.path.join(human_dir, lang)
            if os.path.exists(lang_dir):
                for file in os.listdir(lang_dir):
                    if file.endswith(('.mp3', '.wav', '.flac')):
                        audio_paths.append(os.path.join(lang_dir, file))
                        labels.append(0)
    
    # Collect AI-generated voices (label = 1)
    ai_dir = os.path.join(data_dir, "ai_generated")
    if os.path.exists(ai_dir):
        for lang in ["tamil", "english", "hindi", "malayalam", "telugu"]:
            lang_dir = os.path.join(ai_dir, lang)
            if os.path.exists(lang_dir):
                for file in os.listdir(lang_dir):
                    if file.endswith(('.mp3', '.wav', '.flac')):
                        audio_paths.append(os.path.join(lang_dir, file))
                        labels.append(1)
    
    return audio_paths, labels


def main():
    # Configuration
    DATA_DIR = "./data"  # Change this to your data directory
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Import feature extractor from api.py
    from api import AudioFeatureExtractor
    feature_extractor = AudioFeatureExtractor()
    
    # Collect dataset
    print("Collecting dataset...")
    audio_paths, labels = collect_dataset_paths(DATA_DIR)
    
    if len(audio_paths) == 0:
        print("No audio files found! Please add data to the data directory.")
        print("See PROJECT_PLAN.md for dataset collection instructions.")
        return
    
    print(f"Found {len(audio_paths)} audio samples")
    print(f"Human samples: {labels.count(0)}, AI samples: {labels.count(1)}")
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create datasets
    train_dataset = AudioDataset(train_paths, train_labels, feature_extractor, augment=True)
    val_dataset = AudioDataset(val_paths, val_labels, feature_extractor, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    print("Initializing model...")
    model = ImprovedVoiceClassifier(input_size=109).to(DEVICE)
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), "voice_classifier_best.pth")
    
    # Save model metadata
    metadata = {
        "model_type": "ImprovedVoiceClassifier",
        "input_size": 107,
        "training_samples": len(train_paths),
        "validation_samples": len(val_paths),
        "epochs": EPOCHS,
        "final_metrics": "Check training logs"
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Training complete! Model saved to voice_classifier_best.pth")


if __name__ == "__main__":
    main()
