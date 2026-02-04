"""
AI Voice Sample Generator
Generate AI-generated voice samples for training using Google Text-to-Speech
"""

import os
from pathlib import Path
from gtts import gTTS
import random


# Sample texts for each language
SAMPLE_TEXTS = {
    "english": [
        "Hello, how are you today?",
        "The weather is beautiful this morning.",
        "I enjoy reading books in my free time.",
        "Technology is advancing rapidly every day.",
        "Learning new things makes life interesting.",
        "Music brings joy to many people's lives.",
        "The sun rises in the east every morning.",
        "Kindness is a language everyone understands.",
        "Time is the most valuable resource we have.",
        "Dreams give us hope for the future.",
    ],
    "tamil": [
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "இன்று வானிலை மிக அழகாக உள்ளது.",
        "நான் புத்தகங்கள் படிப்பதை விரும்புகிறேன்.",
        "தொழில்நுட்பம் ஒவ்வொரு நாளும் வேகமாக முன்னேறுகிறது.",
        "புதிய விஷயங்களைக் கற்றுக்கொள்வது வாழ்க்கையை சுவாரஸ்யமாக்குகிறது.",
        "இசை பலருக்கு மகிழ்ச்சியைக் கொடுக்கிறது.",
        "சூரியன் ஒவ்வொரு காலையிலும் கிழக்கில் உதிக்கிறது.",
        "கருணை என்பது அனைவரும் புரிந்துகொள்ளும் மொழி.",
        "நேரம் நம்மிடம் உள்ள மிக மதிப்புமிக்க வளம்.",
        "கனவுகள் நமக்கு எதிர்காலத்தின் மீது நம்பிக்கையை அளிக்கின்றன.",
    ],
    "hindi": [
        "नमस्ते, आप कैसे हैं?",
        "आज का मौसम बहुत सुंदर है।",
        "मुझे अपने खाली समय में किताबें पढ़ना पसंद है।",
        "तकनीक हर दिन तेजी से आगे बढ़ रही है।",
        "नई चीजें सीखना जीवन को दिलचस्प बनाता है।",
        "संगीत कई लोगों के जीवन में खुशी लाता है।",
        "सूरज हर सुबह पूर्व में उगता है।",
        "दयालुता एक ऐसी भाषा है जिसे हर कोई समझता है।",
        "समय हमारे पास सबसे कीमती संसाधन है।",
        "सपने हमें भविष्य के लिए आशा देते हैं।",
    ],
    "malayalam": [
        "ഹലോ, നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?",
        "ഇന്നത്തെ കാലാവസ്ഥ വളരെ മനോഹരമാണ്.",
        "എനിക്ക് ഒഴിവു സമയത്ത് പുസ്തകങ്ങൾ വായിക്കാൻ ഇഷ്ടമാണ്.",
        "സാങ്കേതികവിദ്യ ദിവസേന അതിവേഗം മുന്നേറുകയാണ്.",
        "പുതിയ കാര്യങ്ങൾ പഠിക്കുന്നത് ജീവിതത്തെ രസകരമാക്കുന്നു.",
        "സംഗീതം പലരുടെയും ജീവിതത്തിൽ സന്തോഷം നൽകുന്നു.",
        "സൂര്യൻ ഓരോ പ്രഭാതത്തിലും കിഴക്ക് ഉദിക്കുന്നു.",
        "ദയ എല്ലാവരും മനസ്സിലാക്കുന്ന ഒരു ഭാഷയാണ്.",
        "സമയം നമുക്കുള്ള ഏറ്റവും വിലപ്പെട്ട വിഭവമാണ്.",
        "സ്വപ്നങ്ങൾ നമുക്ക് ഭാവിയിൽ പ്രതീക്ഷ നൽകുന്നു.",
    ],
    "telugu": [
        "హలో, మీరు ఎలా ఉన్నారు?",
        "నేటి వాతావరణం చాలా అందంగా ఉంది.",
        "నాకు ఖాళీ సమయంలో పుస్తకాలు చదవడం చాలా ఇష్టం.",
        "సాంకేతికత ప్రతిరోజూ వేగంగా అభివృద్ధి చెందుతోంది.",
        "కొత్త విషయాలు నేర్చుకోవడం జీవితాన్ని ఆసక్తికరంగా చేస్తుంది.",
        "సంగీతం అనేక మంది జీవితాలలో ఆనందాన్ని తెస్తుంది.",
        "సూర్యుడు ప్రతి ఉదయం తూర్పున ఉదయిస్తాడు.",
        "దయ అనేది అందరూ అర్థం చేసుకునే భాష.",
        "సమయం మనకు ఉన్న అత్యంత విలువైన వనరు.",
        "కలలు మనకు భవిష్యత్తు కోసం ఆశను ఇస్తాయి.",
    ],
}


def generate_gtts_samples(output_dir="data/ai_generated", samples_per_language=50):
    """
    Generate AI voice samples using Google Text-to-Speech (gTTS)
    
    Note: gTTS is free but requires internet connection
    """
    
    print("="*60)
    print("Generating AI Voice Samples using gTTS")
    print("="*60)
    print()
    
    # Language codes for gTTS
    lang_codes = {
        "english": "en",
        "tamil": "ta",
        "hindi": "hi",
        "malayalam": "ml",
        "telugu": "te",
    }
    
    total_generated = 0
    
    for language, texts in SAMPLE_TEXTS.items():
        print(f"Generating {language} samples...")
        
        # Create directory
        lang_dir = Path(output_dir) / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        lang_code = lang_codes[language]
        
        # Generate samples by repeating texts
        for i in range(samples_per_language):
            # Cycle through texts
            text = texts[i % len(texts)]
            
            # Add variations
            if i >= len(texts):
                # Add some variation for repeated texts
                variations = ["", "Indeed, ", "Actually, ", "You know, ", "Well, "]
                text = random.choice(variations) + text
            
            output_path = lang_dir / f"gtts_sample_{i+1:03d}.mp3"
            
            try:
                # Generate TTS
                tts = gTTS(text=text, lang=lang_code, slow=False)
                tts.save(str(output_path))
                
                total_generated += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{samples_per_language} samples")
                    
            except Exception as e:
                print(f"  Error generating sample {i+1}: {str(e)}")
        
        print(f"✓ Completed {language}: {samples_per_language} samples")
        print()
    
    print("="*60)
    print(f"Total samples generated: {total_generated}")
    print("="*60)


def generate_advanced_samples_instructions():
    """
    Show instructions for generating higher-quality AI samples
    """
    print("\n" + "="*60)
    print("Advanced AI Voice Generation (Better Quality)")
    print("="*60)
    print()
    
    print("For better quality AI samples, use these services:")
    print()
    
    print("1. GOOGLE CLOUD TEXT-TO-SPEECH (Recommended)")
    print("   - Higher quality than gTTS")
    print("   - Free tier: 1M characters/month")
    print()
    print("   Setup:")
    print("   pip install google-cloud-texttospeech")
    print("   # Set up Google Cloud credentials")
    print()
    print("   Example code:")
    print('''
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
input_text = texttospeech.SynthesisInput(text="Hello")
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)
response = client.synthesize_speech(
    input=input_text, voice=voice, audio_config=audio_config
)
with open("output.mp3", "wb") as out:
    out.write(response.audio_content)
''')
    
    print()
    print("2. ELEVENLABS (High Quality)")
    print("   - Very realistic AI voices")
    print("   - Free tier: 10k characters/month")
    print("   - Visit: https://elevenlabs.io")
    print()
    
    print("3. MICROSOFT AZURE TTS")
    print("   - Neural voices available")
    print("   - Free tier: 0.5M characters/month")
    print("   - pip install azure-cognitiveservices-speech")
    print()
    
    print("4. COQUI TTS (Open Source)")
    print("   - Run locally, no API limits")
    print("   - pip install TTS")
    print("   - GitHub: https://github.com/coqui-ai/TTS")
    print()


def main():
    """Main function"""
    print("="*60)
    print("AI Voice Sample Generator")
    print("="*60)
    print()
    
    print("This script generates AI voice samples for training.")
    print()
    print("Options:")
    print("1. Generate samples using gTTS (free, requires internet)")
    print("2. View instructions for advanced AI voice generation")
    print("3. Exit")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print()
        samples = input("How many samples per language? (default: 50): ").strip()
        samples = int(samples) if samples else 50
        
        print()
        print("Installing gTTS if needed...")
        os.system("pip install gtts --break-system-packages 2>/dev/null || pip install gtts")
        
        print()
        generate_gtts_samples(samples_per_language=samples)
        
        print()
        print("Next steps:")
        print("1. Collect human voice samples")
        print("2. Run: python train_model.py")
        
    elif choice == "2":
        generate_advanced_samples_instructions()
        
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
