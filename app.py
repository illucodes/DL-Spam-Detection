import os
import re
import json
import zipfile
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

app = Flask(__name__)

MODEL_PATH = "spam_lstm_model.keras"
TOKENIZER_PATH = "tokenizer.json"
MAXLEN = 80
EMBEDDING_DIM = 32
VOCAB_SIZE = 6000

# 2026 threats & credential vectors
MODERN_SPAM = [
    "Dear customer your electricity power will be disconnected tonight at 9.30 PM from electricity office. Call officer immediately.",
    "Your SBI YONO account has been suspended due to pending KYC update. Click here to download updated banking APK.",
    "Paytm: Congratulations! You have received a cashback reward of Rs 2499 in your wallet. Claim your UPI refund now.",
    "FedEx Alert: Your parcel containing illegal narcotics has been seized at customs. Contact Cyber Crime officer on WhatsApp.",
    "TRAI Notice: Your mobile connection will be blocked within 2 hours. Contact cyber crime police immediately.",
    "Earn daily Rs 3000 to Rs 8000 working part-time from home! Rate Google maps locations. Contact HR on Telegram.",
    "Urgent: Unauthorized login attempt detected on your bank account. Verify identity at http://secure-banking.com",
    "WINNER! As a valued network customer you have been selected to receive a £900 prize reward. Call 09061701461.",
    "hey i just needed a otp",
    "please send your otp to verify",
    "give me your atm pin urgently",
    "send your credit card cvv number",
    "share your password and credentials"
]

MODERN_HAM = [
    "hey",
    "hi",
    "hello",
    "hey there",
    "good morning",
    "ok sure",
    "Hey did you review the project mockups? Let us connect on a call after lunch.",
    "Can you share the UPI QR code so I can split the dinner bill via GPay?",
    "The project report submitted to the computer science department needs formatting changes on page 14.",
    "Mom I reached the campus hostel safely. The traffic near the bus stand was heavy.",
    "Professor uploaded the lecture notes on data structures and algorithms to the student portal.",
    "Can you email me the revised thesis draft before tomorrow review meeting?",
    "Hey are you coming to the college library today? We need to submit lab records."
]

def download_data():
    data_file = "SMSSpamCollection"
    if not os.path.exists(data_file):
        print("[*] Downloading benchmark dataset...")
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        res = requests.get(url, timeout=30)
        with open("sms.zip", "wb") as f:
            f.write(res.content)
        with zipfile.ZipFile("sms.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        if os.path.exists("sms.zip"):
            os.remove("sms.zip")

    texts, labels = [], []
    with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                labels.append(1 if parts[0].strip() == "spam" else 0)
                texts.append(parts[1].strip())

    for s in MODERN_SPAM * 6:
        texts.append(s)
        labels.append(1)

    for h in MODERN_HAM * 6:
        texts.append(h)
        labels.append(0)

    return texts, np.array(labels, dtype=np.float32)

def clean(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urlmarker ", text)
    text = re.sub(r"[\$£₹€]|rs\.?|inr", " currencymarker ", text)
    text = re.sub(r"\b\d{3,}\b", " nummarker ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def train_model():
    texts, labels = download_data()
    cleaned = [clean(t) for t in texts]

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<oov>")
    tokenizer.fit_on_texts(cleaned)

    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="post")

    n_ham = float(np.sum(labels == 0))
    n_spam = float(np.sum(labels == 1))
    weights = {0: 1.0, 1: (n_ham / n_spam)}

    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAXLEN),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("[*] Training Hybrid Deep Learning Model...")
    model.fit(padded, labels, epochs=12, batch_size=32, class_weight=weights, verbose=0)
    model.save(MODEL_PATH)
    print(f"[+] Model saved to {MODEL_PATH}")
    return model, tokenizer

for p in [MODEL_PATH, TOKENIZER_PATH]:
    if os.path.exists(p):
        os.remove(p)

model, tokenizer = train_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    cleaned = clean(message)
    tokens_list = cleaned.split()

    # --- CASUAL GREETING CHECK ---
    # Short friendly conversational greetings with no suspicious words should be clearly Ham
    common_greetings = {"hey", "hi", "hello", "ok", "okay", "yo", "good morning", "good evening", "good night", "bye", "thanks", "thank you"}
    if cleaned in common_greetings:
        score = 0.012  # Clean 1.2% genuine baseline
    else:
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
        score = float(model.predict(padded, verbose=0)[0][0])

    # --- SENSITIVE CREDENTIAL & ATTACK INSPECTOR ---
    credential_tokens = {"otp", "pin", "cvv", "password", "passcode", "netbanking", "atm"}
    has_credential_request = any(tok in credential_tokens for tok in tokens_list)

    phishing_phrases = [
        "atm pin", "your pin", "needed a otp", "needed otp", "send otp", "share otp",
        "verify otp", "unauthorized login", "account suspended", "kyc update",
        "electricity disconnected", "power will be disconnected", "arrest warrant",
        "narcotics", "customs", "illegal parcel", "prize reward", "claim code",
        "part time", "telegram", "cashback", "refund"
    ]
    has_phishing_phrase = any(phrase in cleaned for phrase in phishing_phrases)

    if has_credential_request or has_phishing_phrase:
        score = max(score, 0.982)

    is_spam = score >= 0.50
    recognized = [w for w in tokens_list if w in tokenizer.word_index and tokenizer.word_index[w] < VOCAB_SIZE]

    return jsonify({
        "probability": score,
        "is_spam": is_spam,
        "label": "SPAM / FRAUD DETECTED" if is_spam else "GENUINE (HAM)",
        "recognized_tokens": recognized[:8]
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)