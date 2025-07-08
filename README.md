# 🤖 Companion Robot with Smart Reminders and Voice Assistant

This is a **Final Year Project (FYP)** developed using **Python** and **Streamlit**. It functions as a **voice-enabled companion** that can engage in natural conversations, detect user emotions, understand English/Urdu, and schedule intelligent voice-based reminders.

---

## 🎓 Final Year Project Overview

This project demonstrates the integration of modern AI and NLP technologies to build a responsive **Companion Robot**. It combines:

- Natural Language Understanding
- Speech Recognition and Synthesis
- Emotion Detection
- Bilingual Communication (English & Urdu)
- Voice-based Smart Reminders

---

## 💡 Key Features

- 🎙️ **Voice Interaction**: Use your microphone to talk naturally
- 😊 **Emotion Detection**: Understands and reacts to emotional tone
- 🌐 **Bilingual Support**: English and Urdu (auto-detection and translation)
- 🔁 **Conversational Memory**: Maintains chat context across turns
- ⏰ **Reminder System**:
  - Set reminders using natural language (e.g., "Remind me in 10 minutes")
  - Also supports manual time selection
  - Uses voice alerts and optional alarm sounds
- 🎛️ **Microphone Selection**: Choose input device from the sidebar

---

## 📦 Technologies Used

| Category         | Library / Tool                         |
|------------------|----------------------------------------|
| Web Interface    | Streamlit                              |
| Speech to Text   | OpenAI Whisper                         |
| Chatbot Model    | BlenderBot via Hugging Face Transformers |
| Emotion Detection| text2emotion                           |
| Text to Speech   | gTTS, pyttsx3                           |
| Language Tools   | langdetect, deep-translator            |
| Audio Handling   | sounddevice, numpy, scipy, soundfile   |
| System Utility   | FFmpeg (external)                      |

---

## 🛠️ Installation & Setup

### ✅ Prerequisites

- Python 3.8 or higher
- FFmpeg installed and added to your system `PATH`

### 🔧 Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/faryal786-fari/companion-robot.git
   cd companion-robot
Install Dependencies
pip install -r requirements.txt

Add FFmpeg to System PATH

Download FFmpeg: https://ffmpeg.org/download.html

Extract and add the bin folder to your system's environment variables
Example for Windows: C:\ffmpeg\bin

Run the App
streamlit run com_robot.py


## 📁 Project Structure

    ```plaintext
    companion-robot/
    │
    ├── com_robot.py          # Main Streamlit application
    ├── requirements.txt      # Python dependencies
    ├── README.md             # Project documentation
    ├── alarm.mpeg            # Optional alarm sound (convertible to WAV)
    └── assets/               # (Optional) For screenshots or resources


📸 Screenshot 

![image](https://github.com/user-attachments/assets/933ad3cc-0128-4a80-a2b9-7c1f8b7717ba)


🙋‍♀️ About the Author

Name: Faryal Gulzar
Department: Computer Science
University: Rawalpindi Women University
Email: faryalchaudhary9970@gmail.com
GitHub: faryal786-fari

📜 License

This project is intended for academic and educational purposes only.
Do not distribute or use commercially without permission.
