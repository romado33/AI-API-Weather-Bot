# --- Install needed libraries ---
!pip install transformers accelerate requests gradio --quiet

# --- Imports ---
from transformers import pipeline
import requests
import re
import torch
import gradio as gr
from google.colab import userdata

# --- Get weather API key ---
OPENWEATHERMAP_API_KEY = userdata.get("OPENWEATHER_API_KEY")

# --- Use GPU if available ---
device = 0 if torch.cuda.is_available() else -1

# --- Load conversational model (Falcon) ---
chatbot = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    device=device,
    trust_remote_code=True
)

# --- Weather fetcher ---
def get_weather(city):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return f"Could not fetch weather for {city}."

        description = data['weather'][0]['description'].capitalize()
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        return f"{description}, {temp}°C (feels like {feels_like}°C)"
    except Exception as e:
        return f"Weather fetch error: {e}"

# --- Chat logic ---
def chat(city, emotion):
    city = city.strip() or "Ottawa"
    weather = get_weather(city)
    prompt = (
        f"The weather in {city} is: {weather}.\n"
        f"Respond in a {emotion} tone."
    )

    response = chatbot(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        do_sample=True
    )[0]["generated_text"]

    return response[len(prompt):].strip()

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## 🌦️ Ask About the Weather with a Chosen Emotion")
    city_input = gr.Textbox(label="Enter a City", placeholder="e.g., Ottawa")
    emotion_input = gr.Dropdown(
        ["warm", "cheerful", "reassuring", "sarcastic", "dramatic", "scientific", "humorous"],
        label="Choose Response Emotion",
        value="warm"
    )
    output = gr.Textbox(label="Chatbot Response", lines=3)
    submit_btn = gr.Button("Submit")

    submit_btn.click(fn=chat, inputs=[city_input, emotion_input], outputs=output)

demo.launch()
