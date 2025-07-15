# --- Install required libraries ---
!pip install transformers accelerate requests --quiet

# --- Imports ---
from transformers import pipeline
import requests
import re
from google.colab import userdata
import torch

# --- Get API key from Colab secrets ---
OPENWEATHERMAP_API_KEY = userdata.get('OPENWEATHER_API_KEY')  # Set this via userdata.set_secret

# --- Choose device (GPU if available) ---
device = 0 if torch.cuda.is_available() else -1

# --- Load an instruction-tuned conversational model ---
chatbot = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    device=device,
    trust_remote_code=True
)

# --- Extract city name from input using regex ---
def extract_city(user_input):
    match = re.search(r"in ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)", user_input)
    if match:
        return match.group(1)
    return "Ottawa"

# --- Fetch weather data from OpenWeatherMap ---
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

# --- Generate a helpful chatbot reply ---
def chat(user_input):
    city = extract_city(user_input)
    weather = get_weather(city)
    prompt = (
        f"The user asked: '{user_input}'\n"
        f"The weather in {city} is: {weather}.\n"
        f"Respond in a warm, helpful tone."
    )

    response = chatbot(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        do_sample=True
    )[0]['generated_text']

    # Trim output to just the new response (remove original prompt)
    return response[len(prompt):].strip()

# --- Try it out ---
print(chat("What’s the weather like in Ottawa today?"))
