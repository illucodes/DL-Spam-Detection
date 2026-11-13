# Spam Email & SMS Detection System (Deep Learning)

A deep learning web application built with **Python**, **Keras (LSTM)**, and **Flask** that classifies SMS and email messages as **Spam** or **Legitimate (Ham)** in real time.

## Features
* **Deep Learning Model:** Trained an LSTM neural network on the SMS Spam Collection dataset (5,574 labeled messages) using Keras and TensorFlow.
* **Interactive Web Interface:** Multi-page Flask application featuring live message testing, project architecture overview, and evaluation results.
* **Custom Preprocessing Pipeline:** Integrated tokenization (`tokenizer.json`) and sequence padding for real-time inference.

## Tech Stack
* **Backend & ML:** Python, Flask, TensorFlow / Keras, NumPy
* **Frontend:** HTML5, CSS3, JavaScript
* **Dataset:** UCI SMS Spam Collection

## Project Structure
```text
├── app.py                  # Main Flask application & inference routes
├── spam_lstm_model.keras   # Trained LSTM model weights
├── tokenizer.json          # Fitted text tokenizer
├── requirements.txt        # Python dependencies
├── SMSSpamCollection       # Training dataset
└── templates/              # HTML frontend templates
