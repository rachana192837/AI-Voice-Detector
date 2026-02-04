"""
Dataset Collection Script
This script helps you collect and organize audio data for training
"""

import os
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm


def create_directory_structure():
    """Create the required directory structure"""
    base_dir = Path("data")
    
    # Create directories
    for category in ["human", "ai_generated"]:
        for language in ["tamil", "english", "hindi", "malayalam", "telugu"]:
            dir_path = base_dir / category / language
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {dir_path}")
    
    print("\n✓ Directory structure created!")
    return base_dir


def download_common_voice_samples():
    """
    Instructions for downloading Common Voice dataset
    Mozilla Common Voice is a great source for human voice samples
    """
    print("\n" + "="*60)
    print("STEP 1: Download Human Voice Samples")
    print("="*60)
    
    print("\n📌 Common Voice Dataset (Recommended):")
    print("   Visit: https://commonvoice.mozilla.org/en/datasets")
    print("   Languages available: Tamil, English, Hindi, Malayalam")
    print("   Download the datasets and place in data/human/<language>/")
    
    print("\n📌 Alternative Sources:")
    print("   - LibriSpeech (English): https://www.openslr.org/12")
    print("   - VoxForge (Multiple languages)")
    print("   - YouTube Audio Library (with proper licensing)")
    
    print("\n⚠️  Important: Ensure you have rights to use the audio data!")


def generate_ai_voices_instructions():
    """
    Instructions for generating AI voice samples
    """
    print("\n" + "="*60)
    print("STEP 2: Generate AI Voice Samples")
    print("="*60)
    
    print("\n🤖 AI Voice Generation Services:")
    print("\n1. Google Text-to-Speech (Free tier available)")
    print("   - API: https://cloud.google.com/text-to-speech")
    print("   - Supports all 5 languages")
    
    print("\n2. ElevenLabs (Free tier: 10k characters/month)")
    print("   - Website: https://elevenlabs.io")
    print("   - High-quality AI voices")
    
    print("\n3. Microsoft Azure TTS")
    print("   - API: https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/")
    
    print("\n4. Coqui TTS (Open Source)")
    print("   - GitHub: https://github.com/coqui-ai/TTS")
    print("   - Can run locally")
    
    print("\n📝 Script to generate samples:")
    print("   See: generate_ai_samples.py")


def show_sample_generation_code():
    """Show example code for generating AI samples"""
    
    code = '''
# Example: Generate AI samples using Google TTS
from google.cloud import texttospeech
import os

def generate_ai_sample(text, language_code, output_path):
    """Generate AI voice sample using Google TTS"""
    client = texttospeech.TextToSpeechClient()
    
    input_text = texttospeech.SynthesisInput(text=text)
    
    # Select language and voice
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

# Language codes
languages = {
    "tamil": "ta-IN",
    "english": "en-US",
    "hindi": "hi-IN",
    "malayalam": "ml-IN",
    "telugu": "te-IN"
}

# Sample texts (use diverse texts)
sample_texts = [
    "Hello, how are you today?",
    "The weather is nice today.",
    "I love listening to music.",
    # Add more diverse texts
]

# Generate samples
for lang, code in languages.items():
    for i, text in enumerate(sample_texts):
        output_path = f"data/ai_generated/{lang}/sample_{i}.mp3"
        generate_ai_sample(text, code, output_path)
        print(f"Generated: {output_path}")
'''
    
    print("\n" + "="*60)
    print("Sample Generation Code")
    print("="*60)
    print(code)


def verify_dataset():
    """Verify the collected dataset"""
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    base_dir = Path("data")
    
    total_human = 0
    total_ai = 0
    
    print("\n📊 Dataset Statistics:")
    print("-" * 60)
    
    for category in ["human", "ai_generated"]:
        print(f"\n{category.upper()}:")
        for language in ["tamil", "english", "hindi", "malayalam", "telugu"]:
            dir_path = base_dir / category / language
            
            if dir_path.exists():
                files = list(dir_path.glob("*.*"))
                audio_files = [f for f in files if f.suffix in ['.mp3', '.wav', '.flac', '.ogg']]
                count = len(audio_files)
                
                if category == "human":
                    total_human += count
                else:
                    total_ai += count
                
                status = "✓" if count > 0 else "✗"
                print(f"  {status} {language}: {count} samples")
    
    print(f"\n{'='*60}")
    print(f"Total Human Samples: {total_human}")
    print(f"Total AI Samples: {total_ai}")
    print(f"Total Dataset Size: {total_human + total_ai}")
    print(f"{'='*60}")
    
    # Recommendations
    if total_human + total_ai < 100:
        print("\n⚠️  Warning: Dataset is very small (< 100 samples)")
        print("   Recommended minimum: 500 samples per class")
        print("   For best results: 2000+ samples per class")
    elif total_human + total_ai < 1000:
        print("\n⚠️  Dataset is small. Consider adding more samples.")
        print("   Current: {} samples".format(total_human + total_ai))
        print("   Recommended: 1000+ samples per class")
    else:
        print("\n✓ Dataset size looks good!")
    
    # Balance check
    if total_human > 0 and total_ai > 0:
        ratio = max(total_human, total_ai) / min(total_human, total_ai)
        if ratio > 2.0:
            print(f"\n⚠️  Dataset is imbalanced (ratio: {ratio:.2f}:1)")
            print("   Consider balancing the classes for better performance")
        else:
            print(f"\n✓ Dataset is balanced (ratio: {ratio:.2f}:1)")


def main():
    """Main function"""
    print("=" * 60)
    print("AI Voice Detection - Dataset Collection Tool")
    print("=" * 60)
    
    # Create directory structure
    print("\n[1/4] Creating directory structure...")
    create_directory_structure()
    
    # Show download instructions
    print("\n[2/4] Human voice collection...")
    download_common_voice_samples()
    
    # Show AI generation instructions
    print("\n[3/4] AI voice generation...")
    generate_ai_voices_instructions()
    show_sample_generation_code()
    
    # Wait for user to collect data
    input("\n\nPress Enter after you've collected the dataset...")
    
    # Verify dataset
    print("\n[4/4] Verifying dataset...")
    verify_dataset()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. If dataset is ready, run: python train_model.py")
    print("2. After training, test with: python test_api.py")
    print("3. Deploy using: docker build -t voice-detector .")
    print("="*60)


if __name__ == "__main__":
    main()
