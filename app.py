"""
LinguaFlow Flask Backend — Fixed Audio Recognition
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
import base64
import io

app = Flask(__name__)

# ─────────────────────────────────────────────
#  LANGUAGES
# ─────────────────────────────────────────────
LANGUAGES = {
    "en":    ("English",    "en-US"),
    "hi":    ("Hindi",      "hi-IN"),
    "es":    ("Spanish",    "es-ES"),
    "fr":    ("French",     "fr-FR"),
    "de":    ("German",     "de-DE"),
    "it":    ("Italian",    "it-IT"),
    "pt":    ("Portuguese", "pt-BR"),
    "ru":    ("Russian",    "ru-RU"),
    "zh-CN": ("Chinese",    "zh-CN"),
    "ja":    ("Japanese",   "ja-JP"),
    "ko":    ("Korean",     "ko-KR"),
    "ar":    ("Arabic",     "ar-SA"),
    "tr":    ("Turkish",    "tr-TR"),
    "ta":    ("Tamil",      "ta-IN"),
    "te":    ("Telugu",     "te-IN"),
    "kn":    ("Kannada",    "kn-IN"),
    "ml":    ("Malayalam",  "ml-IN"),
    "bn":    ("Bengali",    "bn-BD"),
    "gu":    ("Gujarati",   "gu-IN"),
    "mr":    ("Marathi",    "mr-IN"),
    "pa":    ("Punjabi",    "pa-IN"),
    "ur":    ("Urdu",       "ur-PK"),
    "nl":    ("Dutch",      "nl-NL"),
    "pl":    ("Polish",     "pl-PL"),
    "sv":    ("Swedish",    "sv-SE"),
    "el":    ("Greek",      "el-GR"),
    "th":    ("Thai",       "th-TH"),
    "vi":    ("Vietnamese", "vi-VN"),
    "id":    ("Indonesian", "id-ID"),
    "sw":    ("Swahili",    "sw-KE"),
}

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "").strip()
    src  = data.get("src", "en")
    tgt  = data.get("tgt", "hi")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        if src == tgt:
            return jsonify({"translated": text})
        translated = GoogleTranslator(source=src, target=tgt).translate(text)
        return jsonify({"translated": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return jsonify({"audio": audio_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recognize", methods=["POST"])
def recognize():
    lang_code  = request.form.get("lang", "en-US")
    audio_file = request.files.get("audio")

    if not audio_file:
        return jsonify({"error": "No audio file received"}), 400

    # Save the incoming audio (could be webm, ogg, wav)
    tmp_input  = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_input.close()
    tmp_output.close()

    try:
        # Save uploaded file
        audio_file.save(tmp_input.name)

        # Convert to WAV using pydub (handles webm, ogg, mp4, etc.)
        try:
            audio = AudioSegment.from_file(tmp_input.name)
        except Exception:
            # Try forcing format
            audio = AudioSegment.from_file(tmp_input.name, format="webm")

        # Export as WAV compatible with speech_recognition
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(tmp_output.name, format="wav")

        # Run speech recognition on the WAV file
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_output.name) as source:
            recorded = recognizer.record(source)

        text = recognizer.recognize_google(recorded, language=lang_code)
        return jsonify({"text": text})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand. Please speak clearly."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Recognition service error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Audio processing error: {str(e)}"}), 500
    finally:
        # Clean up temp files
        for f in [tmp_input.name, tmp_output.name]:
            try:
                os.unlink(f)
            except:
                pass


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  LinguaFlow is starting...")
    print("  Open your browser and go to:")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
