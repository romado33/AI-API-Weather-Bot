# --- Install needed libraries ---
!pip install transformers accelerate requests gradio --quiet

# weather_chatbot.py

import requests
import re
import torch
import gradio as gr
from transformers import pipeline
from google.colab import userdata
from functools import lru_cache

# --- API key setup (Colab secrets) ---
OPENWEATHERMAP_API_KEY = userdata.get("OPENWEATHER_API_KEY")

# --- Load small, fast language model (FLAN-T5) ---
device = 0 if torch.cuda.is_available() else -1
chatbot = pipeline("text2text-generation", model="google/flan-t5-base", device=device)

# --- Validate user input ---
def validate_city(city):
    if not re.match(r"^[a-zA-Z\s\-]{2,}$", city):
        return "Ottawa"
    return city.strip()

def validate_range(rng):
    return rng if rng in ["today", "3-day", "7-day"] else "today"

# --- Cache API results to improve speed ---
@lru_cache(maxsize=10)
def get_weather(city, forecast_range):
    city = validate_city(city)
    forecast_range = validate_range(forecast_range)

    try:
        if forecast_range in ["3-day", "7-day"]:
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
            response = requests.get(url, timeout=10)
            data = response.json()

            if response.status_code != 200:
                return f"Could not fetch forecast for {city}."

            entries = data["list"]
            step = 8  # ~1/day
            count = 3 if forecast_range == "3-day" else 7
            forecast = [
                f"📅 {entries[i]['dt_txt'][:10]}: {entries[i]['weather'][0]['description']}, 🌡 {entries[i]['main']['temp']}°C"
                for i in range(0, min(len(entries), count * step), step)
            ]
            return "\n".join(forecast)

        else:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
            response = requests.get(url, timeout=10)
            data = response.json()

            if response.status_code != 200:
                return f"Could not fetch weather for {city}."

            desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            feels = data["main"]["feels_like"]
            return f"{desc}, 🌡 {temp}°C (feels like {feels}°C)"

    except Exception as e:
        return f"Weather fetch error: {e}"

# --- Tone definitions ---
tone_templates = {
    "warm": "Respond in a kind and welcoming tone.",
    "cheerful": "Be upbeat and playful.",
    "reassuring": "Give comfort and optimism.",
    "sarcastic": "Be dry and humorous.",
    "dramatic": "Be theatrical and expressive.",
    "scientific": "Use clear, factual language.",
    "humorous": "Make it witty and fun."
}

# --- History-enabled chat function ---
def chat_with_memory(history, city, emotion, detail, forecast_range):
    city = validate_city(city)
    weather = get_weather(city, forecast_range)
    tone = tone_templates.get(emotion, "Be helpful.")

    prompt = (
        f"City: {city}\n"
        f"Weather: {weather}\n"
        f"Detail: {detail}\n"
        f"Tone: {tone}\n"
        f"Generate a helpful response:\n"
    )

    if len(prompt) > 1000:
        prompt = prompt[:1000]

    response = chatbot(prompt, max_new_tokens=100)[0]["generated_text"].strip()
    history.append((f"Weather in {city} ({forecast_range}) with a {emotion} tone:", response))
    return history, history

# --- Gradio UI with chat log ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌦️ AI Weather Chatbot")

    with gr.Row():
        with gr.Column():
            city = gr.Textbox(label="Enter a City", placeholder="e.g., Ottawa")
            emotion = gr.Dropdown(["warm", "cheerful", "reassuring", "sarcastic", "dramatic", "scientific", "humorous"], value="warm", label="Response Tone")
            detail = gr.Dropdown(["brief", "standard", "detailed"], value="standard", label="Level of Detail")
            forecast_range = gr.Dropdown(["today", "3-day", "7-day"], value="today", label="Forecast Range")
            submit_btn = gr.Button("Get Forecast")
            clear_btn = gr.Button("Clear History")

        with gr.Column():
            chatbot_output = gr.Chatbot(label="Chat History")
            state = gr.State([])

    submit_btn.click(fn=chat_with_memory, inputs=[state, city, emotion, detail, forecast_range], outputs=[chatbot_output, state])
    clear_btn.click(fn=lambda: ([], []), inputs=[], outputs=[chatbot_output, state])

demo.launch(debug=True)

