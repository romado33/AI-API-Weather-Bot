---
title: AI API Weather Bot
emoji: 📄
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 5.37.0
python_version: '3.10'
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
---

This project is a small Gradio demo that fetches weather information and presents it in the style of famous movie characters.

## Running locally

1. Install dependencies with `pip install -r requirements.txt`.
2. Set the environment variable `OPENWEATHER_API_KEY` with your OpenWeatherMap key.
3. Launch the application using `python app.py`.

Without an API key the app falls back to randomly generated forecast data.

