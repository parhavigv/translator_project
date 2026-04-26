# ── STEP 2: Test your microphone and speech recognition ──
# Save as: test_mic.py
# Run with: python test_mic.py

import speech_recognition as sr

print("=== Microphone + Speech Recognition Test ===\n")

# List all available mics
print("Available microphones on your system:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"  [{i}] {name}")

print("\n--- Starting mic test ---")
print("Speak something after 'Listening...' appears.")
print("(You have 5 seconds)\n")

recognizer = sr.Recognizer()

try:
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=1)
        print("🎙️  Listening... speak now!")
        audio = recognizer.listen(mic, timeout=5, phrase_time_limit=5)

    print("⏳ Recognizing...")
    text = recognizer.recognize_google(audio, language="en-US")
    print(f"\n✅ SUCCESS! You said: \"{text}\"")
    print("\nMicrophone is working. Run: python test_translate.py")

except sr.WaitTimeoutError:
    print("❌ No speech detected. Check your mic is plugged in and try again.")
except sr.UnknownValueError:
    print("❌ Could not understand. Speak louder and more clearly.")
except sr.RequestError as e:
    print(f"❌ Internet error: {e}")
    print("   Make sure you are connected to the internet.")
except OSError as e:
    print(f"❌ Mic hardware error: {e}")
    print("   Make sure your microphone is connected and not in use by another app.")
