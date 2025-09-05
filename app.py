"""Gradio-based weather app with character-themed forecast responses."""

from datetime import datetime, timedelta
import os
import random
from typing import Dict, List, Tuple

import gradio as gr
import plotly.graph_objects as go
import requests


OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Mapping from UI temperature choices to OpenWeather API units
TEMP_UNIT_TO_API = {"Celsius": "metric", "Fahrenheit": "imperial"}

CHARACTER_TEMPLATES = {
    "Napoleon Dynamite": lambda txt: f"Gosh. {txt} This is like the worst. Idiot.",
    "Colonel Nathan Jessup": lambda txt: f"You want the forecast? YOU CAN'T HANDLE THE WEATHER! {txt}",
    "Yoda": lambda txt: f"{txt} Hmm. Forecast strong with the Force, it is.",
    "Tony Stark": lambda txt: f"{txt} Built it in a cave... with a box of scraps! Just kidding. Weather tech on point.",
    "Forrest Gump": lambda txt: f"Mama always said the weather is like a box of chocolates. {txt}",
    "The Dude": lambda txt: f"Yeah well, {txt}. The Dude abides.",
    "Terminator": lambda txt: f"{txt}. I'll be back... with more data.",
    "Gandalf": lambda txt: f"You shall not pass... without knowing: {txt}",
    "Austin Powers": lambda txt: f"Yeah baby! {txt} Shagadelic weather update!",
    "Deadpool": lambda txt: f"Here's your f***ing forecast: {txt}. You're welcome.",
}


def _random_forecast() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, str]]:
    """Return random forecast data when the real API is unavailable."""

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    hourly = [
        {
            "time": (now + timedelta(hours=i)).strftime("%H:%M"),
            "temp_c": random.randint(-10, 35),
        }
        for i in range(12)
    ]
    daily = [
        {
            "date": (now + timedelta(days=i)).strftime("%Y-%m-%d"),
            "max_temp_c": random.randint(-5, 35),
            "min_temp_c": random.randint(-15, 20),
        }
        for i in range(5)
    ]
    current = {
        "temp_c": random.randint(-10, 35),
        "description": random.choice(
            ["Clear skies", "Cloudy", "Light rain", "Snow", "Windy"]
        ),
    }
    return hourly, daily, current


def fetch_forecast_data(
    city: str, temp_unit: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, str], bool]:
    """Fetch forecast data and indicate whether it's real or demo.

    Args:
        city: Name of the city to look up.
        temp_unit: "Celsius" or "Fahrenheit" to control the units used when
            querying the OpenWeather API.

    Returns:
        Tuple containing hourly data, daily data, current conditions and a
        boolean flag indicating whether the values are randomly generated
        (True) or fetched from the API (False).
    """

    city = city.strip()
    if not city or not OPENWEATHER_API_KEY:
        hourly, daily, current = _random_forecast()
        return hourly, daily, current, True

    # Convert the user-facing unit string to the API's expected value.
    units = TEMP_UNIT_TO_API.get(temp_unit, "metric")

    try:
        geo_res = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": city, "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=10,
        )
        geo_res.raise_for_status()
        coords = geo_res.json()
        if not coords:
            hourly, daily, current = _random_forecast()
            return hourly, daily, current, True
        lat, lon = coords[0]["lat"], coords[0]["lon"]

        weather_res = requests.get(
            "https://api.openweathermap.org/data/2.5/onecall",
            params={
                "lat": lat,
                "lon": lon,
                "exclude": "minutely,alerts",
                "units": units,
                "appid": OPENWEATHER_API_KEY,
            },
            timeout=10,
        )
        weather_res.raise_for_status()
        data = weather_res.json()
    except (requests.RequestException, ValueError):
        hourly, daily, current = _random_forecast()
        return hourly, daily, current, True

    def to_c(temp: float) -> int:
        """Convert temperature to Celsius if API returned Fahrenheit."""

        return round((temp - 32) * 5 / 9) if units == "imperial" else round(temp)

    hourly = [
        {
            "time": datetime.fromtimestamp(h["dt"]).strftime("%H:%M"),
            "temp_c": to_c(h["temp"]),
        }
        for h in data.get("hourly", [])[:12]
    ]
    daily = [
        {
            "date": datetime.fromtimestamp(d["dt"]).strftime("%Y-%m-%d"),
            "max_temp_c": to_c(d["temp"]["max"]),
            "min_temp_c": to_c(d["temp"]["min"]),
        }
        for d in data.get("daily", [])[:5]
    ]
    current = {
        "temp_c": to_c(data.get("current", {}).get("temp", 0)),
        "description": data.get("current", {})
        .get("weather", [{}])[0]
        .get("description", "")
        .title(),
    }
    return hourly, daily, current, False


def create_current_conditions_chart(current: Dict[str, str], temp_unit: str) -> go.Figure:
    """Create a simple chart summarizing current conditions."""

    def convert(temp: int) -> int:
        return round(temp * 9 / 5 + 32) if temp_unit == "Fahrenheit" else temp

    unit_symbol = "F" if temp_unit == "Fahrenheit" else "C"
    fig = go.Figure()
    fig.add_annotation(
        text=f"{convert(current['temp_c'])}°{unit_symbol} - {current['description']}",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=24),
        xref="paper",
        yref="paper",
    )
    fig.update_layout(
        title="Current Conditions", xaxis=dict(visible=False), yaxis=dict(visible=False), height=200
    )
    return fig


def create_hourly_forecast_chart(hourly_forecast: List[Dict[str, str]], temp_unit: str) -> go.Figure:
    def convert(temp: int) -> int:
        return round(temp * 9 / 5 + 32) if temp_unit == "Fahrenheit" else temp

    times = [h["time"] for h in hourly_forecast]
    temps = [convert(h["temp_c"]) for h in hourly_forecast]
    y_label = "Temp (F)" if temp_unit == "Fahrenheit" else "Temp (C)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=times, y=temps, mode="lines+markers", name="Temp", line=dict(color="royalblue"))
    )
    fig.update_layout(title="Hourly Forecast", xaxis_title="Time", yaxis_title=y_label, height=400)
    return fig


def create_daily_forecast_chart(daily_forecast: List[Dict[str, str]], temp_unit: str) -> go.Figure:
    def convert(temp: int) -> int:
        return round(temp * 9 / 5 + 32) if temp_unit == "Fahrenheit" else temp

    dates = [d["date"] for d in daily_forecast]
    max_temps = [convert(d["max_temp_c"]) for d in daily_forecast]
    min_temps = [convert(d["min_temp_c"]) for d in daily_forecast]
    y_label = "Temp (F)" if temp_unit == "Fahrenheit" else "Temp (C)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=max_temps, name="Max", mode="lines+markers", line=dict(color="red"))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=min_temps, name="Min", mode="lines+markers", line=dict(color="blue"))
    )
    fig.update_layout(title="5-Day Forecast", xaxis_title="Date", yaxis_title=y_label, height=400)
    return fig



def generate_character_response(
    city: str,
    forecast_range: str,
    character: str,
    hourly_data: List[Dict[str, str]] | None = None,
    daily_data: List[Dict[str, str]] | None = None,
    current_data: Dict[str, str] | None = None,
    temp_unit: str = "Celsius",
) -> str:
    """Create a character-styled weather report."""

    def convert(temp: int) -> int:
        return round(temp * 9 / 5 + 32) if temp_unit == "Fahrenheit" else temp

    unit_symbol = "F" if temp_unit == "Fahrenheit" else "C"

    city_intro = f"The weather in {city}..."
    if forecast_range == "Today" and current_data:
        lines = [
            f"Currently {convert(current_data['temp_c'])}°{unit_symbol} with {current_data['description'].lower()}."
        ]
    elif forecast_range == "Hourly" and hourly_data:
        lines = [
            f"At {h['time']}, it's {convert(h['temp_c'])}°{unit_symbol}." for h in hourly_data
        ]
    elif forecast_range == "5-Day" and daily_data:
        lines = [
            f"{d['date']}: High {convert(d['max_temp_c'])}°{unit_symbol} / Low {convert(d['min_temp_c'])}°{unit_symbol}."
            for d in daily_data
        ]
    else:
        lines = []

    report = city_intro + ("\\n" + " ".join(lines) if lines else "")
    return CHARACTER_TEMPLATES.get(character, lambda x: x)(report)



def character_weather_chat(history, city, character, forecast_range, temp_unit, session_id):
    hourly_data, daily_data, current_data, is_demo = fetch_forecast_data(city, temp_unit)
    response = generate_character_response(
        city, forecast_range, character, hourly_data, daily_data, current_data, temp_unit
    )
    warning = "Using demo data; set OPENWEATHER_API_KEY for live weather" if is_demo else ""
    full_response = response + ("\n\n" + warning if warning else "")
    user_query = f"{character or 'Character'} - Weather for {city}"
    new_history = history or []
    new_history.append({"role": "user", "content": user_query})
    new_history.append({"role": "assistant", "content": full_response})

    if forecast_range == "Today" and current_data:
        chart = create_current_conditions_chart(current_data, temp_unit)
    elif forecast_range == "Hourly" and hourly_data:
        chart = create_hourly_forecast_chart(hourly_data, temp_unit)
    elif forecast_range == "5-Day" and daily_data:
        chart = create_daily_forecast_chart(daily_data, temp_unit)
    else:
        chart = None

    return new_history, new_history, chart


try:  # Dummy GPU decorator for HF Spaces
    import spaces

    @spaces.GPU
    def init_dummy():
        return True
except ImportError:  # pragma: no cover - optional dependency
    pass


with gr.Blocks(title="🎬 Character Weather Forecast") as demo:
    gr.Markdown("## 🎭 Movie Character Weather App with Forecasts")

    with gr.Row():
        city_input = gr.Textbox(label="City", value="Ottawa")
        temp_unit = gr.Radio(choices=["Celsius", "Fahrenheit"], value="Celsius", label="Unit")
        forecast_range = gr.Radio(choices=["Today", "Hourly", "5-Day"], value="Today", label="Forecast Range")

    with gr.Row():
        character_select = gr.Radio(
            choices=list(CHARACTER_TEMPLATES.keys()),
            label="Character",
        )

    get_weather_btn = gr.Button("Get Character Weather")

    with gr.Row():
        chat_display = gr.Chatbot(label="Forecast Chat", type="messages")
        chart_output = gr.Plot(label="Forecast Chart")

    state = gr.State([])

    get_weather_btn.click(
        fn=character_weather_chat,
        inputs=[state, city_input, character_select, forecast_range, temp_unit, gr.State(str(random.randint(100000, 999999)))],
        outputs=[chat_display, state, chart_output],
    )


if __name__ == "__main__":
    demo.launch()

