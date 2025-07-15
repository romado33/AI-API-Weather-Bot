# AI---Colab---API-Weather-Bot# 🌤️ AI-Powered Weather Chatbot

This is an interactive, AI-powered weather chatbot that gives you **personalized, real-time weather reports** — in a tone you choose!

## 🧠 What it does

- Asks the user to input:
  - A city name
  - A tone/emotion (warm, sarcastic, scientific, etc.)
  - A forecast range (today / 3-day / 7-day)
  - A level of detail (brief / standard / detailed)
- Fetches real-time weather data from [OpenWeatherMap](https://openweathermap.org/)
- Uses a conversational LLM (like Falcon-RW-1B or FLAN-T5) to generate a custom chatbot-style response
- Built entirely in Python, using Gradio and Hugging Face Transformers

## 🔗 Live Demo

👉 [Open in Google Colab](https://colab.research.google.com/github/yourusername/weather-chatbot-ai/blob/main/weather_chatbot.ipynb)

## 🛠 Tech Stack

- [Gradio](https://www.gradio.app/) – for the interactive web UI
- [Transformers](https://huggingface.co/docs/transformers/index) – for text generation
- [OpenWeatherMap API](https://openweathermap.org/api) – for live weather data
- [Google Colab](https://colab.research.google.com/) – for free, hosted execution

## ⚙️ Setup

1. Clone the repo or open the notebook in Colab
2. Set your OpenWeatherMap API key using:
   ```python
   from google.colab import userdata
   userdata.set_secret('OPENWEATHERMAP_API_KEY')


