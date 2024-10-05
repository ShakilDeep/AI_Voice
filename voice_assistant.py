import os
import sys
import pyaudio
import wave
import requests
import groq
import pyttsx3
from dotenv import load_dotenv


def load_api_keys():
    load_dotenv()
    deepgram_key = os.getenv('DEEPGRAM_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')

    if not deepgram_key:
        deepgram_key = input("Please enter your Deepgram API key: ").strip()
    if not groq_key:
        groq_key = input("Please enter your Groq API key: ").strip()

    return deepgram_key, groq_key


# Load API keys
DEEPGRAM_API_KEY, GROQ_API_KEY = load_api_keys()

# Validate API keys
if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    print("Error: Both Deepgram and Groq API keys are required.")
    sys.exit(1)

# Initialize Groq client
try:
    groq_client = groq.Client(api_key=GROQ_API_KEY)
except groq.GroqError as e:
    print(f"Error initializing Groq client: {e}")
    print("Please check your Groq API key and ensure it's correct.")
    sys.exit(1)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Listening... Speak now.")
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
    print("Processing...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def transcribe_audio():
    try:
        with open(WAVE_OUTPUT_FILENAME, 'rb') as audio:
            response = requests.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/wav"
                },
                data=audio
            )
        response.raise_for_status()
        return response.json()['results']['channels'][0]['alternatives'][0]['transcript']
    except requests.RequestException as e:
        print(f"Error transcribing audio: {e}")
        return None


def get_groq_response(transcript):
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a helpful voice assistant. Provide concise and relevant responses."},
                {"role": "user", "content": transcript}
            ],
            model="mixtral-8x7b-32768",
            max_tokens=200,
        )
        return completion.choices[0].message.content
    except groq.GroqError as e:
        print(f"Error getting response from Groq: {e}")
        return None


def speak_response(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


def main():
    print("Voice Assistant is ready. Say 'exit' to quit.")
    while True:
        record_audio()
        transcript = transcribe_audio()
        if transcript:
            print(f"You said: {transcript}")

            if transcript.lower() == 'exit':
                print("Exiting the voice assistant. Goodbye!")
                speak_response("Exiting the voice assistant. Goodbye!")
                break

            response = get_groq_response(transcript)
            if response:
                print(f"Assistant: {response}")
                speak_response(response)
            else:
                error_msg = "I'm sorry, I couldn't generate a response. Please try again."
                print(error_msg)
                speak_response(error_msg)
        else:
            error_msg = "I'm sorry, I couldn't understand that. Could you please repeat?"
            print(error_msg)
            speak_response(error_msg)


if __name__ == "__main__":
    main()