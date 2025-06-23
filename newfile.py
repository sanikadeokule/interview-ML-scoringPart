import torchaudio
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from deep_translator import GoogleTranslator
import torch
from pydub import AudioSegment
import os
from flask import Flask, request, jsonify
from threading import Lock

app = Flask(__name__)

# Globals for lazy loading
whisper = None
tokenizer = None
model = None
model_lock = Lock()

def load_models():
    global whisper, tokenizer, model
    with model_lock:
        if whisper is None:
            whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        if tokenizer is None or model is None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_audio_transcription(audio_path):
    load_models()
    result = whisper(audio_path)
    return result['text']

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def get_ai_rating(question, answer):
    load_models()
    inputs = tokenizer(question, answer, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits)[0].item()
        return round(score * 100, 2)  # Return score out of 100

def convert_to_wav_if_needed(input_path):
    if input_path.lower().endswith(".wav"):
        return input_path  # No need to convert

    output_path = os.path.splitext(input_path)[0] + ".wav"
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"🎧 Converted to WAV: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error converting to WAV: {e}")
        return None

def get_question_from_backend():
    # Replace this with your backend logic (DB/config fetch)
    return "What is binary search?"

# ...existing code...

@app.route("/", methods=["GET"])
def home():
    return "API is running. Use POST / with an audio file (key: audio)."

@app.route("/", methods=["POST"])
def evaluate():
    # Get audio file from frontend
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    audio_file = request.files["audio"]
    temp_path = "temp_uploaded_audio"
    audio_file.save(temp_path)

    # Convert to wav if needed
    wav_path = convert_to_wav_if_needed(temp_path)
    if not wav_path:
        return jsonify({"error": "Could not convert audio"}), 500

    # Get question from backend
    question = get_question_from_backend()

    # Process
    spoken_text = get_audio_transcription(wav_path)
    translated_answer = translate_to_english(spoken_text)
    rating = get_ai_rating(question, translated_answer)

    # Clean up temp files
    try:
        os.remove(temp_path)
        if wav_path != temp_path:
            os.remove(wav_path)
    except Exception:
        pass

    return jsonify({
        "question": question,
        "transcribed": spoken_text,
        "translated": translated_answer,
        "rating": rating
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)