import requests
import re
import torch
import gradio as gr
from transformers import pipeline
from functools import lru_cache
import os

# --- Load API key securely ---
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "demo_key")

# --- Load FLAN-T5 model (small, fast) ---
device = 0 if torch.cuda.is_available() else -1
chatbot = pipeline("text2text-generation", model="google/flan-t5-base", device=device)

# --- Tone template map ---
tone_templates = {
    "warm": "a kind and welcoming",
    "cheerful": "an upbeat and playful",
    "reassuring": "a calming and optimistic",
    "sarcastic": "a dry and witty",
    "dramatic": "a theatrical and expressive",
    "scientific": "a factual and precise",
    "humorous": "a funny and lighthearted"
}

# --- Input validation ---
def validate_city(city):
    if not re.match(r"^[a-zA-Z\s\-]{2,}$", city):
        return "Ottawa"
    return city.strip()

def validate_range(rng):
    return rng if rng in ["today", "3-day", "7-day"] else "today"

# --- Weather fetch with cache ---
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

# --- Chat logic with improved prompt ---
def chat_with_memory(history, city, emotion, detail, forecast_range):
    city = validate_city(city)
    weather = get_weather(city, forecast_range)
    tone_desc = tone_templates.get(emotion, "helpful")

    prompt = (
        f"The current weather in {city} is: {weather}.\n"
        f"Please write a {detail} weather update in {tone_desc} tone."
    )

    if len(prompt) > 1000:
        prompt = prompt[:1000]

    response = chatbot(prompt, max_new_tokens=120)[0]["generated_text"].strip()

    history.append((f"{city} ({forecast_range}, {emotion}, {detail})", response))
    return history, history

# --- Gradio app layout ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌦️ AI Weather Chatbot – Personalized Forecasts")

    with gr.Row():
        with gr.Column():
            city = gr.Textbox(label="Enter a City", placeholder="e.g., Ottawa")
            emotion = gr.Dropdown(["warm", "cheerful", "reassuring", "sarcastic", "dramatic", "scientific", "humorous"], value="warm", label="Tone of Voice")
            detail = gr.Dropdown(["brief", "standard", "detailed"], value="standard", label="Level of Detail")
            forecast_range = gr.Dropdown(["today", "3-day", "7-day"], value="today", label="Forecast Range")
            submit_btn = gr.Button("Get Forecast")
            clear_btn = gr.Button("Clear Chat")

        with gr.Column():
            chatbox = gr.Chatbot(label="Chat History")
            state = gr.State([])

    submit_btn.click(fn=chat_with_memory, inputs=[state, city, emotion, detail, forecast_range], outputs=[chatbox, state])
    clear_btn.click(fn=lambda: ([], []), inputs=[], outputs=[chatbox, state])

demo.launch()
