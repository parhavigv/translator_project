# ── STEP 1: Run this first to check all libraries are installed ──
# Save as: test_setup.py
# Run with: python test_setup.py

print("Testing all required libraries...\n")

# Test 1
try:
    import speech_recognition as sr
    print("✅ SpeechRecognition - OK")
except ImportError:
    print("❌ SpeechRecognition missing  →  pip install SpeechRecognition")

# Test 2
try:
    from deep_translator import GoogleTranslator
    print("✅ deep-translator - OK")
except ImportError:
    print("❌ deep-translator missing  →  pip install deep-translator")

# Test 3
try:
    import pyttsx3
    print("✅ pyttsx3 - OK")
except ImportError:
    print("❌ pyttsx3 missing  →  pip install pyttsx3")

# Test 4
try:
    import pyaudio
    print("✅ pyaudio - OK")
except ImportError:
    print("❌ pyaudio missing  →  pip install pyaudio")

print("\nIf all show ✅ run:  python test_mic.py")
