import streamlit as st
import datetime
import time
import threading
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import os
import subprocess
import whisper
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS
import text2emotion as te
import re
import nltk
import pyttsx3
import base64
nltk.download('punkt', quiet=True)

# Set page
st.set_page_config(page_title="Companion Robot", page_icon="🤖", layout="centered")
st.title("🤖 Companion Robot with Smart Reminders + Voice Assistant")
# 🔄 Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active" not in st.session_state:
    st.session_state.active = False

if "reminders" not in st.session_state:
    st.session_state.reminders = []

if "thread_started" not in st.session_state:
    st.session_state.thread_started = False

from streamlit_autorefresh import st_autorefresh

# Auto refresh every 5 seconds
st_autorefresh(interval=5000, key="reminder_refresh")


# 📌 Ensure ffmpeg
FFMPEG_PATH = r"C:\\ffmpeg\\bin"
if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH
try:
    subprocess.check_output(["ffmpeg", "-version"])
except FileNotFoundError:
    st.error("❌ ffmpeg not found!")
    st.stop()

# 📌 Audio device selection
devices = sd.query_devices()
input_devices = {f"{i}: {d['name']}": i for i, d in enumerate(devices) if d['max_input_channels'] > 0}

with st.sidebar:
    st.subheader("🎙️ Select Microphone")
    selected_device_label = st.selectbox("Choose device:", list(input_devices.keys()))
    selected_device_index = input_devices[selected_device_label]
    st.markdown(f"✅ Selected: {selected_device_label}")

# 📌 Load models
@st.cache_resource
def load_models():
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    whisper_model = whisper.load_model("base")
    return tokenizer, model, whisper_model

tokenizer, bot_model, whisper_model = load_models()
#for voice conversion
def convert_mpeg_to_wav(input_path="alarm.mpeg", output_path="alarm.wav"):
  
# 📌 TTS engine for reminders (non-bot)
  engine= pyttsx3.init()
  engine.setProperty('rate', 150)

# 📌 Reminder background checker
def reminder_checker():
    while True:
        now = datetime.datetime.now()
        for reminder in st.session_state.reminders:
            if not reminder.get("notified") and reminder["time"] <= now:
                speak_response(f"⏰ Reminder: {reminder['message']}", lang='en')
                reminder["notified"] = True
        time.sleep(1)

# Start once
if not st.session_state.thread_started:
    threading.Thread(target=reminder_checker, daemon=True).start()
    st.session_state.thread_started = True

# 📌 Helper functions
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)
#mpeg file ko convert  karna
def convert_mpeg_to_wav(input_path="alarm.mpeg", output_path="alarm.wav"):
    try:
        subprocess.run(["ffmpeg", "-i", input_path, output_path], check=True)
        print("✅ Conversion successful!")
    except subprocess.CalledProcessError:
        st.error("❌ Failed to convert MPEG to WAV")


def detect_emotion(text):
    emotions = te.get_emotion(text)
    return max(emotions, key=emotions.get) if emotions else "neutral"

def generate_response(user_input, emotion):
    st.session_state.chat_history.append(f"🧑 You: {user_input}")
    context = "\n".join(st.session_state.chat_history[-5:])
    prompt = f"User is feeling [{emotion}]. {context}"
    inputs = tokenizer([prompt], return_tensors="pt")
    reply_ids = bot_model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    st.session_state.chat_history.append(f"🤖 Bot: {response}")
    return response

def speak_response(text, lang='en', emotion='neutral'):
    if lang == 'ur':
        text += "!" if emotion in ["happy", "surprise"] else "."
    elif emotion == "angry":
        text = "😠 " + text.upper()
    elif emotion == "sad":
        text = "😢 " + text.lower()
    elif emotion == "happy":
        text = "😊 " + text.capitalize()

    tts = gTTS(text=text, lang='ur' if lang == 'ur' else 'en')
    tts.save("response.mp3")

    with open("response.mp3", "rb") as f:
        audio_data = f.read()
        b64 = base64.b64encode(audio_data).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        #for beep
def play_beep(duration=5, freq=440):
    fs = 44100  # Sampling frequency
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    beep = 0.5 * np.sin(2 * np.pi * freq * t)
    sd.play(beep, fs)
    sd.wait()
    #for my own voice
import soundfile as sf

def play_custom_alarm(sound_path="alarm.wav"):
    try:
        data, fs = sf.read(sound_path, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        st.error(f"❌ Alarm playback failed: {e}")


def record_until_silence(threshold=0.003, silence_duration=1.5, fs=16000, device_index=None):
    buffer_duration = 0.2
    buffer_size = int(fs * buffer_duration)
    silence_chunks_required = int(silence_duration / buffer_duration)

    if device_index is not None:
        sd.default.device = (device_index, None)

    st.info("🎤 Listening... Speak now.")
    audio_data = []
    silence_counter = 0
    voice_triggered = False

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
            while st.session_state.active:
                buffer, _ = stream.read(buffer_size)
                buffer = np.squeeze(buffer)
                rms = np.sqrt(np.mean(buffer**2))

                if rms > threshold:
                    voice_triggered = True
                    audio_data.append(buffer)
                    silence_counter = 0
                elif voice_triggered:
                    audio_data.append(buffer)
                    silence_counter += 1
                    if silence_counter >= silence_chunks_required:
                        break

        if not voice_triggered:
            st.warning("⚠️ No valid speech detected.")
            return None

        st.success("✅ Done listening.")
        audio_np = np.concatenate(audio_data)
        audio_np = (audio_np * 32767).astype(np.int16)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(temp_wav.name, fs, audio_np)
        return temp_wav.name
    except Exception as e:
        st.error(f"❌ Recording error: {e}")
        return None

def recognize_speech(audio_file):
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        st.error(f"❌ Whisper failed: {e}")
        return ""

def handle_conversation():
    wav_path = record_until_silence(device_index=selected_device_index)
    if not wav_path:
        return

    user_text = recognize_speech(wav_path)
    user_text_clean = remove_emojis(user_text.strip())
    if not user_text_clean:
        return

    try:
        lang = detect(user_text_clean)
    except:
        lang = "en"

    translated_input = GoogleTranslator(source='auto', target='en').translate(user_text_clean) if lang == 'ur' else user_text_clean
    emotion = detect_emotion(translated_input)
    response = generate_response(translated_input, emotion)

    final_response = GoogleTranslator(source='en', target='ur').translate(response) if lang == 'ur' else response

    st.subheader("🗣️ You said:")
    st.write(user_text_clean)
    st.subheader("😊 Detected Emotion:")
    st.write(emotion.capitalize())
    st.subheader("🤖 Bot Response:")
    st.write(final_response)

    speak_response(final_response, lang, emotion)

    st.subheader("💬 Chat History")
    for msg in st.session_state.chat_history:
        st.write(msg)

# 📌 Buttons for listening
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ Start Listening"):
        st.session_state.active = True
        while st.session_state.active:
            handle_conversation()
with col2:
    if st.button("🛑 Stop Listening"):
        st.session_state.active = False
        st.success("🛑 Listening stopped.")

# 📌 Natural Language Reminder Parsing
def parse_natural_language_time(text):
    match = re.search(r"in (\d+) (second|seconds|minute|minutes|hour|hours)", text.lower())
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta = datetime.timedelta(
            seconds=amount if "second" in unit else 0,
            minutes=amount if "minute" in unit else 0,
            hours=amount if "hour" in unit else 0
        )
        return datetime.datetime.now() + delta
    return None

# 📌 Sidebar Reminder Section
st.sidebar.subheader("⏰ Set Reminder")

reminder_input = st.sidebar.text_input("Natural Language (e.g. 'Remind me in 10 minutes')")
reminder_msg = st.sidebar.text_input("Reminder Message", "Take a break!")

date = st.sidebar.date_input("Pick a Date", value=datetime.datetime.now().date())
time_input = st.sidebar.time_input("Pick a Time", value=(datetime.datetime.now() + datetime.timedelta(minutes=1)).time())
selected_datetime = datetime.datetime.combine(date, time_input)

if st.sidebar.button("Set Reminder"):
    if reminder_input.strip():
        parsed_time = parse_natural_language_time(reminder_input)
        if parsed_time:
            st.session_state.reminders.append({"time": parsed_time, "message": reminder_msg, "notified": False})
            st.sidebar.success(f"Reminder set for {parsed_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.sidebar.warning("⚠️ Couldn't parse time. Using selected date/time.")
            st.session_state.reminders.append({"time": selected_datetime, "message": reminder_msg, "notified": False})
            st.sidebar.success(f"Reminder set for {selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.session_state.reminders.append({"time": selected_datetime, "message": reminder_msg, "notified": False})
        st.sidebar.success(f"Reminder set for {selected_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# 📌 Active Reminders Display
st.subheader("📋 Active Reminders")
for reminder in st.session_state.reminders:
    if not reminder.get("notified"):
        st.write(f"🔔 {reminder['message']} at {reminder['time'].strftime('%Y-%m-%d %H:%M:%S')}")
# 📌 Reminder Triggering Logic (run every refresh)
# 📌 Reminder Triggering Logic (run every refresh)
now = datetime.datetime.now()
for reminder in st.session_state.reminders:
    if not reminder.get("notified") and reminder["time"] <= now:
        speak_response(f"⏰ Reminder: {reminder['message']}", lang='en')
        time.sleep(1)  # 1 sec ruk jao taake voice complete ho
        play_custom_alarm("alarm.wav")
        reminder["notified"] = True