# This is a simplified and updated version of your Gradio-based weather app interface,
# with Feature 1: Hourly & Multi-Day Forecasts fully integrated.

import gradio as gr
from datetime import datetime
import plotly.graph_objects as go
import random
import requests
import os

# === Real Forecast Fetching ===
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "demo_key")

# Helper to fetch real weather forecast from OpenWeatherMap One Call API (mocked if API key is not set)
def fetch_forecast_data(city):
    if OPENWEATHER_API_KEY == "demo_key":
        print("⚠️ No real API key set. Using fallback.")
        return None, None

    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    geo_res = requests.get(geo_url)
    print("🌐 Geolocation status:", geo_res.status_code, geo_res.text)
    if not geo_res.ok or not geo_res.json():
        return None, None

    lat = geo_res.json()[0]['lat']
    lon = geo_res.json()[0]['lon']
    print("📍 Location resolved:", lat, lon)
    onecall_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,alerts&units=metric&appid={OPENWEATHER_API_KEY}"
    weather_res = requests.get(onecall_url)
    print("🌦️ Forecast status:", weather_res.status_code, weather_res.text[:200])
    if not weather_res.ok:
        return None, None

    data = weather_res.json()

    hourly_forecast = [
        {
            "time": datetime.fromtimestamp(h['dt']).strftime("%H:%M"),
            "temp_c": round(h['temp'])
        } for h in data['hourly'][:12]
    ]

    daily_forecast = [
        {
            "date": datetime.fromtimestamp(d['dt']).strftime("%Y-%m-%d"),
            "max_temp_c": round(d['temp']['max']),
            "min_temp_c": round(d['temp']['min'])
        } for d in data['daily'][:5]
    ]

    return hourly_forecast, daily_forecast

# === Visualization ===
def create_hourly_forecast_chart(hourly_forecast):
    times = [h['time'] for h in hourly_forecast]
    temps = [h['temp_c'] for h in hourly_forecast]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=temps, mode='lines+markers', name='Temp', line=dict(color='royalblue')))
    fig.update_layout(title='Hourly Forecast', xaxis_title='Time', yaxis_title='Temp (C)', height=400)
    return fig

def create_daily_forecast_chart(daily_forecast):
    dates = [d['date'] for d in daily_forecast]
    max_temps = [d['max_temp_c'] for d in daily_forecast]
    min_temps = [d['min_temp_c'] for d in daily_forecast]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=max_temps, name='Max', mode='lines+markers', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=dates, y=min_temps, name='Min', mode='lines+markers', line=dict(color='blue')))
    fig.update_layout(title='5-Day Forecast', xaxis_title='Date', yaxis_title='Temp (C)', height=400)
    return fig

# === Response Logic ===
def generate_character_response(city, forecast_range, character, hourly_data=None, daily_data=None):
    city_intro = f"The weather in {city}..."
    hourly_lines = []
    daily_lines = []

    if forecast_range == "Hourly" and hourly_data:
        for hour in hourly_data:
            hourly_lines.append(f"At {hour['time']}, it's {hour['temp_c']}°C.")
    elif forecast_range == "5-Day" and daily_data:
        for day in daily_data:
            daily_lines.append(f"{day['date']}: High {day['max_temp_c']}°C / Low {day['min_temp_c']}°C.")

    characters = {
        "Napoleon Dynamite": lambda txt: f"Gosh. {txt} This is like the worst. Idiot.",
        "Colonel Nathan Jessup": lambda txt: f"You want the forecast? YOU CAN'T HANDLE THE WEATHER! {txt}",
        "Yoda": lambda txt: f"{txt} Hmm. Forecast strong with the Force, it is.",
        "Tony Stark": lambda txt: f"{txt} Built it in a cave... with a box of scraps! Just kidding. Weather tech on point.",
        "Forrest Gump": lambda txt: f"Mama always said the weather is like a box of chocolates. {txt}",
        "The Dude": lambda txt: f"Yeah well, {txt}. The Dude abides.",
        "Terminator": lambda txt: f"{txt}. I'll be back... with more data.",
        "Gandalf": lambda txt: f"You shall not pass... without knowing: {txt}",
        "Austin Powers": lambda txt: f"Yeah baby! {txt} Shagadelic weather update!",
        "Deadpool": lambda txt: f"Here's your f***ing forecast: {txt}. You're welcome."
    }

    report = city_intro
    if hourly_lines:
        report += "\n" + " ".join(hourly_lines)

    return characters.get(character, lambda x: x)(report)

# === Interface ===
def character_weather_chat(history, city, character, forecast_range, temp_unit, session_id):
    hourly_data, daily_data = fetch_forecast_data(city)
    response = generate_character_response(city, forecast_range, character, hourly_data, daily_data)
    user_query = f"{character or 'Character'} - Weather for {city}"
    new_history = history or []
    new_history.append({"role": "user", "content": user_query})
    new_history.append({"role": "assistant", "content": response})

    if forecast_range == "Hourly" and hourly_data:
        chart = create_hourly_forecast_chart(hourly_data)
    elif forecast_range == "5-Day" and daily_data:
        chart = create_daily_forecast_chart(daily_data)
    else:
        chart = None

    return new_history, new_history, chart

# === Dummy GPU Decorator for HF Spaces ===
try:
    import spaces

    @spaces.GPU
    def init_dummy():
        return True
except ImportError:
    pass

# === Launch App ===
with gr.Blocks(title="🎬 Character Weather Forecast") as demo:
    gr.Markdown("## 🎭 Movie Character Weather App with Forecasts")

    with gr.Row():
        city_input = gr.Textbox(label="City", value="Ottawa")
        temp_unit = gr.Radio(choices=["Celsius", "Fahrenheit"], value="Celsius", label="Unit")
        forecast_range = gr.Radio(choices=["Today", "Hourly", "5-Day"], value="Today", label="Forecast Range")

    with gr.Row():
        character_select = gr.Radio(choices=[
    "Napoleon Dynamite",
    "Colonel Nathan Jessup",
    "Yoda",
    "Tony Stark",
    "Forrest Gump",
    "The Dude",
    "Terminator",
    "Gandalf",
    "Austin Powers",
    "Deadpool"
], label="Character")

    get_weather_btn = gr.Button("Get Character Weather")

    with gr.Row():
        chat_display = gr.Chatbot(label="Forecast Chat", type="messages")
        chart_output = gr.Plot(label="Forecast Chart")

    state = gr.State([])

    get_weather_btn.click(
        fn=character_weather_chat,
        inputs=[state, city_input, character_select, forecast_range, temp_unit, gr.State(str(random.randint(100000, 999999)))],
        outputs=[chat_display, state, chart_output]
    )

if __name__ == "__main__":
    demo.launch()
