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
            try:
                whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
                print("✅ Whisper model loaded.")
            except Exception as e:
                print(f"❌ Whisper load error: {e}")
        if tokenizer is None or model is None:
            try:
                model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print("✅ Cross-encoder model loaded.")
            except Exception as e:
                print(f"❌ Model/tokenizer load error: {e}")

def get_audio_transcription(audio_path):
    load_models()
    print(f"🔍 Transcribing audio: {audio_path}")
    result = whisper(audio_path)
    print(f"📝 Transcription result: {result['text']}")
    return result['text']

def translate_to_english(text):
    print(f"🌐 Translating: {text}")
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    print(f"➡️ Translated to English: {translated}")
    return translated

def get_ai_rating(question, answer):
    load_models()
    print(f"🎯 Getting AI score for:\nQ: {question}\nA: {answer}")
    inputs = tokenizer(question, answer, return_tensors='pt', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits)[0].item()
        print(f"📊 Raw Score: {score}")
        return round(score * 100, 2)

def convert_to_wav_if_needed(input_path):
    if input_path.lower().endswith(".wav"):
        print("✅ Already WAV format.")
        return input_path

    output_path = os.path.splitext(input_path)[0] + ".wav"
    try:
        print("🎧 Converting audio to WAV using pydub...")
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"✅ Conversion successful: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error during audio conversion: {e}")
        return None

def get_question_from_backend():
    return "What is binary search?"

def get_question_domain(question):
    q = question.lower()
    if any(word in q for word in ["search", "sort", "array", "tree", "algorithm", "complexity"]):
        return "Data Structures & Algorithms"
    if any(word in q for word in ["database", "sql", "table", "query"]):
        return "Databases"
    if any(word in q for word in ["python", "java", "c++", "language", "code", "programming"]):
        return "Programming"
    if any(word in q for word in ["network", "protocol", "ip", "tcp", "udp"]):
        return "Networking"
    if any(word in q for word in ["os", "operating system", "process", "thread", "memory"]):
        return "Operating Systems"
    if any(word in q for word in ["ml", "machine learning", "model", "training", "ai"]):
        return "Machine Learning"
    if any(word in q for word in ["web", "html", "css", "javascript", "frontend", "backend"]):
        return "Web Development"
    if any(word in q for word in ["cloud", "aws", "azure", "gcp", "deployment"]):
        return "Cloud Computing"
    if any(word in q for word in ["capital", "country", "city"]):
        return "General Knowledge"
    return "Other"

@app.route("/", methods=["GET"])
def home():
    return "✅ API is running. Use POST / with form-data key: 'audio'."

@app.route("/", methods=["POST"])
def evaluate():
    print("📩 POST request received.")
    if "audio" not in request.files:
        print("❌ No audio file found in request.")
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    temp_path = "temp_uploaded_audio"
    audio_file.save(temp_path)
    print(f"📁 Audio file saved at: {temp_path}")

    wav_path = convert_to_wav_if_needed(temp_path)
    if not wav_path:
        print("❌ Failed to convert audio to WAV.")
        return jsonify({"error": "Could not convert audio"}), 500

    question = get_question_from_backend()
    domain = get_question_domain(question)
    print(f"❓ Question: {question} | 🏷️ Domain: {domain}")

    spoken_text = get_audio_transcription(wav_path)
    translated_answer = translate_to_english(spoken_text)
    rating = get_ai_rating(question, translated_answer)

    # Clean up
    try:
        os.remove(temp_path)
        if wav_path != temp_path:
            os.remove(wav_path)
        print("🧹 Temp files cleaned.")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

    return jsonify({
        "question": question,
        "domain": domain,
        "transcribed": spoken_text,
        "translated": translated_answer,
        "rating": rating
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
