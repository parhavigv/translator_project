"""
LinguaFlow -- Speech to Speech Translator
Uses gTTS (Google Text-to-Speech) for proper voice output
in ALL languages including Tamil, Kannada, Hindi etc.

INSTALL (run once):
    pip install SpeechRecognition deep-translator pyaudio gtts pygame
"""

import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame
import tempfile
import os
import sys
import time

# ─────────────────────────────────────────────
#  LANGUAGES  (name, translate-code, speech-recognition-code, gtts-code)
# ─────────────────────────────────────────────
LANGUAGES = {
    "1":  ("English",    "en",    "en-US",  "en"),
    "2":  ("Hindi",      "hi",    "hi-IN",  "hi"),
    "3":  ("Spanish",    "es",    "es-ES",  "es"),
    "4":  ("French",     "fr",    "fr-FR",  "fr"),
    "5":  ("German",     "de",    "de-DE",  "de"),
    "6":  ("Italian",    "it",    "it-IT",  "it"),
    "7":  ("Portuguese", "pt",    "pt-BR",  "pt"),
    "8":  ("Russian",    "ru",    "ru-RU",  "ru"),
    "9":  ("Chinese",    "zh-CN", "zh-CN",  "zh-CN"),
    "10": ("Japanese",   "ja",    "ja-JP",  "ja"),
    "11": ("Korean",     "ko",    "ko-KR",  "ko"),
    "12": ("Arabic",     "ar",    "ar-SA",  "ar"),
    "13": ("Turkish",    "tr",    "tr-TR",  "tr"),
    "14": ("Tamil",      "ta",    "ta-IN",  "ta"),
    "15": ("Telugu",     "te",    "te-IN",  "te"),
    "16": ("Kannada",    "kn",    "kn-IN",  "kn"),
    "17": ("Malayalam",  "ml",    "ml-IN",  "ml"),
    "18": ("Bengali",    "bn",    "bn-BD",  "bn"),
    "19": ("Gujarati",   "gu",    "gu-IN",  "gu"),
    "20": ("Marathi",    "mr",    "mr-IN",  "mr"),
    "21": ("Punjabi",    "pa",    "pa-IN",  "pa"),
    "22": ("Urdu",       "ur",    "ur-PK",  "ur"),
    "23": ("Dutch",      "nl",    "nl-NL",  "nl"),
    "24": ("Polish",     "pl",    "pl-PL",  "pl"),
    "25": ("Swedish",    "sv",    "sv-SE",  "sv"),
    "26": ("Greek",      "el",    "el-GR",  "el"),
    "27": ("Thai",       "th",    "th-TH",  "th"),
    "28": ("Vietnamese", "vi",    "vi-VN",  "vi"),
    "29": ("Indonesian", "id",    "id-ID",  "id"),
    "30": ("Swahili",    "sw",    "sw-KE",  "sw"),
}

# ─────────────────────────────────────────────
#  INIT PYGAME AUDIO (used to play gTTS audio)
# ─────────────────────────────────────────────
pygame.mixer.init()

# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────

def banner():
    print("\n" + "="*56)
    print("   LinguaFlow -- Speech to Speech Translator")
    print("   Google Voices | 30 Languages | 100% Free")
    print("="*56)

def show_languages():
    print("\n  Languages:")
    print("  " + "-"*50)
    items = list(LANGUAGES.items())
    for i in range(0, len(items), 2):
        left  = f"  [{items[i][0]:>2}] {items[i][1][0]:<14}"
        right = f"  [{items[i+1][0]:>2}] {items[i+1][1][0]}" if i+1 < len(items) else ""
        print(left + right)
    print("  " + "-"*50)

def pick_language(prompt):
    show_languages()
    while True:
        choice = input(f"\n  {prompt} - enter number: ").strip()
        if choice in LANGUAGES:
            name, trans_code, speech_code, gtts_code = LANGUAGES[choice]
            print(f"  OK: {name} selected")
            return name, trans_code, speech_code, gtts_code
        print("  Invalid. Try again.")

# ─────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────

def listen(speech_code, lang_name):
    """Capture mic and return recognized text, or None on failure."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 1.2
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as mic:
            print(f"\n  Listening in {lang_name}...")
            print("  Speak now. Pause when done.\n")
            recognizer.adjust_for_ambient_noise(mic, duration=0.8)
            audio = recognizer.listen(mic, timeout=10, phrase_time_limit=15)
    except sr.WaitTimeoutError:
        print("  No speech detected. Try again.")
        return None
    except OSError as e:
        print(f"  Microphone error: {e}")
        return None

    print("  Recognizing...")
    try:
        text = recognizer.recognize_google(audio, language=speech_code)
        print(f"  Heard: \"{text}\"")
        return text
    except sr.UnknownValueError:
        print("  Could not understand. Speak clearly and try again.")
        return None
    except sr.RequestError:
        print("  Internet error during recognition. Check your connection.")
        return None


def translate(text, src_code, tgt_code):
    """Translate text using Google Translate (free). Returns string or None."""
    if src_code == tgt_code:
        return text
    print("  Translating...")
    try:
        result = GoogleTranslator(source=src_code, target=tgt_code).translate(text)
        print(f"  Translated: \"{result}\"")
        return result
    except Exception as e:
        print(f"  Translation failed: {e}")
        return None


def speak(text, lang_name, gtts_code):
    """
    Speak text using gTTS (Google Text-to-Speech).
    Supports ALL languages including Tamil, Kannada, Hindi etc.
    Plays audio via pygame.
    """
    print(f"\n  Speaking ({lang_name}): {text}")
    try:
        # Generate speech audio using Google TTS
        tts = gTTS(text=text, lang=gtts_code, slow=False)

        # Save to a temporary mp3 file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.close()
        tts.save(tmp.name)

        # Play the mp3 using pygame
        pygame.mixer.music.load(tmp.name)
        pygame.mixer.music.play()

        # Wait until finished playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # Cleanup temp file
        pygame.mixer.music.unload()
        os.unlink(tmp.name)
        print("  Done speaking.\n")

    except Exception as e:
        print(f"  Speaking failed: {e}")
        print("  Check your internet connection (gTTS needs internet).")


# ─────────────────────────────────────────────
#  MODES
# ─────────────────────────────────────────────

def mode_speech_to_speech():
    print("\n" + "-"*56)
    print("  SPEECH TO SPEECH TRANSLATION")
    print("-"*56)
    src_name, src_trans, src_speech, _        = pick_language("Language you will SPEAK")
    tgt_name, tgt_trans, _,          tgt_gtts = pick_language("Language you want to HEAR")

    print(f"\n  Ready! {src_name} -> {tgt_name}")
    print("  Press Enter to speak, type q + Enter to stop.\n")

    while True:
        cmd = input("  [Enter = speak]  [q = back to menu]: ").strip().lower()
        if cmd == 'q':
            break

        # Step 1: Listen
        heard = listen(src_speech, src_name)
        if not heard:
            continue

        # Step 2: Translate
        translated = translate(heard, src_trans, tgt_trans)
        if not translated:
            continue

        print(f"\n  You said    ({src_name}): {heard}")
        print(f"  Translation ({tgt_name}): {translated}")

        # Step 3: Speak with Google voice
        speak(translated, tgt_name, tgt_gtts)


def mode_speech_to_text():
    print("\n" + "-"*56)
    print("  SPEECH TO TEXT")
    print("-"*56)
    src_name, src_trans, src_speech, _ = pick_language("Language you will SPEAK")
    tgt_name, tgt_trans, _, _          = pick_language("Language to TRANSLATE into")

    heard = listen(src_speech, src_name)
    if not heard:
        return

    translated = translate(heard, src_trans, tgt_trans)
    if translated:
        print(f"\n  Original   ({src_name}): {heard}")
        print(f"  Translated ({tgt_name}): {translated}")


def mode_text_to_text():
    print("\n" + "-"*56)
    print("  TEXT TO TEXT TRANSLATION")
    print("-"*56)
    src_name, src_trans, _, _ = pick_language("SOURCE language")
    tgt_name, tgt_trans, _, _ = pick_language("TARGET language")
    text = input(f"\n  Type {src_name} text: ").strip()
    if not text:
        print("  Nothing entered.")
        return

    translated = translate(text, src_trans, tgt_trans)
    if translated:
        print(f"\n  Original   ({src_name}): {text}")
        print(f"  Translated ({tgt_name}): {translated}")


def mode_text_to_speech():
    print("\n" + "-"*56)
    print("  TEXT TO SPEECH")
    print("-"*56)
    src_name, _, _, gtts_code = pick_language("Language of your text")
    text = input(f"\n  Type text to speak: ").strip()
    if not text:
        print("  Nothing entered.")
        return
    speak(text, src_name, gtts_code)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    banner()
    print("\n  Audio system ready (using Google voices).")

    while True:
        print("\n" + "="*56)
        print("  SELECT MODE")
        print("-"*56)
        print("  [1]  Speech to Speech Translation")
        print("  [2]  Speech to Text  (+ translate)")
        print("  [3]  Text  to Text Translation")
        print("  [4]  Text  to Speech")
        print("  [q]  Quit")
        print("-"*56)
        choice = input("  Your choice: ").strip().lower()

        if   choice == '1': mode_speech_to_speech()
        elif choice == '2': mode_speech_to_text()
        elif choice == '3': mode_text_to_text()
        elif choice == '4': mode_text_to_speech()
        elif choice == 'q':
            print("\n  Goodbye!\n")
            pygame.mixer.quit()
            sys.exit(0)
        else:
            print("  Enter 1, 2, 3, 4 or q.")


if __name__ == "__main__":
    main()
