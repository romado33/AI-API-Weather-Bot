# 🌤️ AI Weather Chatbot

An interactive chatbot that gives personalized, real-time weather forecasts with your choice of tone and detail — powered by OpenWeatherMap, Hugging Face Transformers, and Gradio.

---

## 🚀 Live Demo

👉 Try it on [Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/weather-chatbot)  
🎯 Built with: `flan-t5-base`, Gradio UI, OpenWeatherMap API

---

## Features

- 🌍 **City selection**
- 🤖 **LLM-generated responses** (choose tone: warm, sarcastic, scientific…)
- 📅 **Forecast options** (today, 3-day, 7-day)
- 📏 **Detail control** (brief, standard, detailed)
- 💬 **Chat history with memory**

---

## Screenshots

> Add a screenshot like this:


yaml
Copy
Edit

---

## Setup

To run locally or in a Space:

1. Create an `.env` variable on Hugging Face with your OpenWeatherMap key:
   - Name: `OPENWEATHERMAP_API_KEY`
2. Install dependencies:
pip install -r requirements.txt

markdown
Copy
Edit
3. Run the app:
python app.py

yaml
Copy
Edit

---

## Credits

Created by [Rob Dods](https://www.https://www.linkedin.com/in/rob-dods/) 
Weather data: [OpenWeatherMap](https://openweathermap.org/)  
Language model: [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)

---

