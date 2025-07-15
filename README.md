# 🌤️ AI-Powered Weather Chatbot

This is an interactive, AI-powered weather chatbot that delivers **real-time forecasts** with **custom emotional tones and detail levels**. Built in Python using OpenWeatherMap, Hugging Face Transformers, and Gradio — and runs entirely inside Google Colab.

## 🧠 Features

- 🌍 **City Selection** – Users can input any city worldwide
- 😄 **Emotional Tone Options** – Choose from warm, sarcastic, dramatic, scientific, etc.
- 📆 **Forecast Range** – Get weather for today, 3-day, or 7-day forecast
- 📏 **Detail Level** – Choose between brief, standard, or detailed responses
- 🤖 **LLM-Powered Chat** – Friendly, natural-sounding replies from small transformer models like `falcon-rw-1b` or `flan-t5-base`

## 🚀 Try It Live

- 👉 [Open in Google Colab](https://colab.research.google.com/github/yourusername/weather-chatbot-ai/blob/main/weather_chatbot.ipynb)
- 🗺️ Shareable via Gradio public link (temporary during Colab session)

## 🛠 Tech Stack

| Component        | Description                                    |
|------------------|------------------------------------------------|
| `Gradio`         | Interactive UI (dropdowns, textboxes, etc.)    |
| `Transformers`   | LLMs for generating conversational output      |
| `OpenWeatherMap` | Real-time weather data                         |
| `Google Colab`   | Free cloud runtime + GPU support               |

## 📦 Installation & Setup

1. Open the [Colab notebook](https://colab.research.google.com/github/yourusername/weather-chatbot-ai/blob/main/weather_chatbot.ipynb)
2. Set your OpenWeatherMap API key:
   ```python
   from google.colab import userdata
   userdata.set_secret('OPENWEATHERMAP_API_KEY')
Run all cells and interact with the chatbot through the Gradio interface


