import streamlit as st
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3
import requests

# Configure Gemini API
GEMINI_API_KEY = "..."  # Replace with your Gemini API key

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speech rate
engine.setProperty("volume", 1.0)  # Volume

def record_audio(duration=5, samplerate=44100):
    """
    Record audio using sounddevice and return recognized text.
    - duration: Recording duration in seconds.
    - samplerate: Sampling rate for recording.
    """
    st.info(f"Recording for {duration} seconds...")
    try:
        # Record audio using sounddevice
        audio_data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32",
        )
        sd.wait()  # Wait for recording to finish
        st.success("Recording complete. Processing audio...")

        # Convert NumPy array to bytes for speech recognition
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        recognizer = sr.Recognizer()
        audio_file = sr.AudioData(audio_bytes, samplerate, 2)  # Convert to AudioData object

        # Perform speech-to-text using recognizer
        return recognizer.recognize_google(audio_file)
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while recording audio: {e}")
        return None

def query_gemini(prompt):
    """Send a prompt to Gemini LLM using the API key and return the response."""
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "max_output_tokens": 150,  # Adjust based on your requirement
    }
    try:
        response = requests.post("https://gemini.googleapis.com/v1/generate", headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx
        data = response.json()
        return data.get("output", "No response received from Gemini.")
    except requests.RequestException as e:
        st.error(f"Error connecting to Gemini API: {e}")
        return "I'm having trouble connecting to the AI right now."

def speak_text(text):
    """Convert text to speech and play it back."""
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("Voice AI Assistant")
st.markdown("### Speak to interact with the AI")

# Record audio and process
if st.button("Record and Process"):
    user_input = record_audio(duration=5)  # Record for 5 seconds
    if user_input:
        st.success(f"You said: {user_input}")
        response = query_gemini(user_input)
        st.success(f"AI Response: {response}")
        if st.button("Play AI Response"):
            speak_text(response)
    else:
        st.warning("No valid input detected.")
