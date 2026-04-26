# ── STEP 4: Test text-to-speech (speaker output) ──
# Save as: test_tts.py
# Run with: python test_tts.py

import pyttsx3

print("=== Text-to-Speech Test ===\n")
print("You should hear your computer speak in 3 seconds...")

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    # List available voices
    voices = engine.getProperty('voices')
    print(f"Available voices on your system: {len(voices)}")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v.name} — {v.languages}")

    print("\n🔊 Speaking now...")
    engine.say("Hello! Text to speech is working correctly. Your translator is ready.")
    engine.runAndWait()
    print("\n✅ TTS working! Now run: python translator_app.py")

except Exception as e:
    print(f"❌ TTS failed: {e}")
    print("   On Linux run: sudo apt-get install espeak")
    print("   On Mac/Windows this should work automatically.")
