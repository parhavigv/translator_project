# ── STEP 3: Test translation ──
# Save as: test_translate.py
# Run with: python test_translate.py

from deep_translator import GoogleTranslator

print("=== Translation Test ===\n")

tests = [
    ("Hello, how are you?",   "en", "hi", "English → Hindi"),
    ("Good morning everyone", "en", "es", "English → Spanish"),
    ("Thank you very much",   "en", "fr", "English → French"),
    ("I love Python",         "en", "de", "English → German"),
]

all_ok = True
for text, src, tgt, label in tests:
    try:
        result = GoogleTranslator(source=src, target=tgt).translate(text)
        print(f"✅ {label}")
        print(f"   Input : {text}")
        print(f"   Output: {result}\n")
    except Exception as e:
        print(f"❌ {label} FAILED: {e}\n")
        all_ok = False

if all_ok:
    print("All translations working! Run: python test_tts.py")
else:
    print("Some translations failed. Check internet connection.")
