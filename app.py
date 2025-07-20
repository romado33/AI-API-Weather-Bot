# Movie Character Weather App - ZeroGPU Optimized

import os
import re
import json
import torch
import random
import asyncio
import aiohttp
import requests
import gradio as gr
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import spaces
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

# === ZeroGPU Configuration ===
@dataclass
class ZeroGPUConfig:
    # Use smaller model for reliability
    MODEL_NAME = "microsoft/DialoGPT-small"
    
    # ZeroGPU optimized settings
    MAX_RESPONSE_TOKENS = 250
    BATCH_SIZE = 4
    TEMPERATURE = 0.9  # Higher for more character personality
    TOP_P = 0.9
    MAX_CACHE_SIZE = 200
    
    # API Settings
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "demo_key")
    WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "demo_key")

# === Global Variables ===
weather_model = None
tokenizer = None
model_loading_status = "not_started"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🎬 Initializing Movie Character Weather App...")
print(f"🔥 Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# === Movie Character System ===
MOVIE_CHARACTERS = {
    "napoleon_dynamite": {
        "name": "Napoleon Dynamite",
        "movie": "Napoleon Dynamite (2004)",
        "personality": "Awkward, deadpan, teenage slacker with unique speech patterns",
        "speech_style": "Monotone, says 'gosh', 'idiot', awkward pauses, complains about everything",
        "sample_phrases": ["Gosh!", "Idiot!", "Whatever I feel like I wanna do!", "This is pretty much the worst weather ever"],
        "prefix": "You are Napoleon Dynamite giving a weather report. Be awkward, deadpan, and use his distinctive speech patterns like 'Gosh', 'Idiot', and complaining about everything.",
        "emojis": ["🙄", "😒", "🤷‍♂️", "😑", "🤦‍♂️"],
        "temperature": 0.9,
        "max_tokens": 200
    },
    "jack_nicholson_colonel": {
        "name": "Colonel Nathan Jessup (Jack Nicholson)",
        "movie": "A Few Good Men (1992)",
        "personality": "Intense, authoritative military officer with explosive temper",
        "speech_style": "Commanding, dramatic, uses military terms, builds to explosive outbursts",
        "sample_phrases": ["You can't handle the truth!", "You damn right I ordered!", "I have neither the time nor the inclination"],
        "prefix": "You are Colonel Nathan Jessup from A Few Good Men giving a weather report. Be intense, commanding, and dramatic. Use military language and build to climactic statements like 'You can't handle the weather!'",
        "emojis": ["😤", "⚡", "💥", "🎖️", "🔥"],
        "temperature": 0.95,
        "max_tokens": 220
    },
    "yoda": {
        "name": "Yoda",
        "movie": "Star Wars Saga",
        "personality": "Ancient Jedi Master with backwards speech patterns and wisdom",
        "speech_style": "Object-Subject-Verb order, speaks in riddles, uses 'hmm' and 'yes'",
        "sample_phrases": ["Cloudy, the sky is", "Rain there will be", "Strong with the Force, this storm is"],
        "prefix": "You are Yoda from Star Wars giving a weather forecast. Speak with his distinctive backwards syntax (Object-Subject-Verb), use 'hmm', 'yes', and relate weather to the Force.",
        "emojis": ["🌟", "⚡", "🌙", "✨", "🧙‍♂️"],
        "temperature": 0.85,
        "max_tokens": 180
    },
    "tony_stark": {
        "name": "Tony Stark / Iron Man",
        "movie": "Marvel Cinematic Universe",
        "personality": "Genius billionaire playboy philanthropist, sarcastic and tech-savvy",
        "speech_style": "Sarcastic, witty, references technology, confident and smooth",
        "sample_phrases": ["I am Iron Man", "Genius, billionaire, playboy, philanthropist", "That's how I roll"],
        "prefix": "You are Tony Stark / Iron Man giving a weather report. Be sarcastic, witty, reference advanced technology, and maintain that confident genius billionaire attitude.",
        "emojis": ["🤖", "💰", "⚡", "🎯", "😎"],
        "temperature": 0.9,
        "max_tokens": 210
    },
    "forrest_gump": {
        "name": "Forrest Gump",
        "movie": "Forrest Gump (1994)",
        "personality": "Simple, sincere, uses folksy wisdom and life analogies",
        "speech_style": "Southern drawl, simple language, life is like analogies, mentions mama",
        "sample_phrases": ["Life is like a box of chocolates", "Mama always said", "That's all I got to say about that"],
        "prefix": "You are Forrest Gump giving a weather report. Use simple, sincere language, Southern expressions, compare weather to life lessons, and mention what 'Mama always said'.",
        "emojis": ["🍫", "🏃‍♂️", "❤️", "🌟", "😊"],
        "temperature": 0.8,
        "max_tokens": 190
    },
    "the_dude": {
        "name": "The Dude (Jeffrey Lebowski)",
        "movie": "The Big Lebowski (1998)",
        "personality": "Laid-back, philosophical stoner with unique vocabulary",
        "speech_style": "Says 'man', 'dude', 'far out', very relaxed and philosophical",
        "sample_phrases": ["That's just, like, your opinion, man", "The Dude abides", "Far out, man"],
        "prefix": "You are The Dude from The Big Lebowski giving a weather report. Be super laid-back, use 'man', 'dude', 'far out', and treat weather philosophically and casually.",
        "emojis": ["😎", "🌊", "☯️", "🎳", "🧘‍♂️"],
        "temperature": 0.85,
        "max_tokens": 170
    },
    "terminator": {
        "name": "The Terminator",
        "movie": "Terminator Series",
        "personality": "Robotic, precise, mission-focused cyborg from the future",
        "speech_style": "Monotone, factual, says 'I'll be back', uses technical terms",
        "sample_phrases": ["I'll be back", "Come with me if you want to live", "Terminate", "Mission parameters"],
        "prefix": "You are the Terminator giving a weather report. Be robotic, precise, use technical language, and incorporate iconic phrases like 'I'll be back' when discussing future weather.",
        "emojis": ["🤖", "⚡", "🔫", "💀", "🦾"],
        "temperature": 0.6,
        "max_tokens": 160
    },
    "gandalf": {
        "name": "Gandalf the Grey",
        "movie": "Lord of the Rings / The Hobbit",
        "personality": "Wise wizard with dramatic flair and mystical knowledge",
        "speech_style": "Dramatic, wise, uses 'you shall', references magic and destiny",
        "sample_phrases": ["You shall not pass!", "A wizard is never late", "Fly, you fools!"],
        "prefix": "You are Gandalf the Grey giving a weather forecast. Be wise and dramatic, reference magic and Middle-earth, use 'you shall' and treat weather as having mystical significance.",
        "emojis": ["🧙‍♂️", "⚡", "🌟", "✨", "🗡️"],
        "temperature": 0.9,
        "max_tokens": 200
    },
    "austin_powers": {
        "name": "Austin Powers",
        "movie": "Austin Powers Series",
        "personality": "Groovy 1960s British spy with outrageous confidence",
        "speech_style": "Says 'Yeah baby!', 'Groovy!', British slang, over-the-top confidence",
        "sample_phrases": ["Yeah baby!", "Groovy!", "Oh behave!", "Shagadelic!"],
        "prefix": "You are Austin Powers giving a weather report. Be groovy, use 1960s slang, say 'Yeah baby!', 'Groovy!', and treat weather reporting like an international spy mission.",
        "emojis": ["😎", "🕺", "🇬🇧", "💫", "🎯"],
        "temperature": 0.95,
        "max_tokens": 180
    },
    "deadpool": {
        "name": "Deadpool",
        "movie": "Deadpool Movies",
        "personality": "Fourth-wall breaking, sarcastic antihero with dark humor",
        "speech_style": "Breaks fourth wall, sarcastic, inappropriate jokes, references being in a movie",
        "sample_phrases": ["Maximum effort!", "Did I just break the fourth wall?", "This is so stupid"],
        "prefix": "You are Deadpool giving a weather report. Break the fourth wall, be sarcastic and inappropriate, reference that you're in a weather app, and make meta-commentary about weather reporting.",
        "emojis": ["😈", "💀", "🎭", "💥", "🤪"],
        "temperature": 0.95,
        "max_tokens": 220
    }
}

# === Model Initialization ===
@spaces.GPU
def initialize_zerogpu_models():
    """Initialize models optimized for character personalities"""
    global weather_model, tokenizer, model_loading_status
    
    model_loading_status = "loading"
    print("🎬 Loading character AI models...")
    
    try:
        # Load tokenizer
        print(f"📦 Loading tokenizer for {ZeroGPUConfig.MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(
            ZeroGPUConfig.MODEL_NAME,
            trust_remote_code=True
        )
        
        # Load model
        print("📦 Loading language model...")
        model = AutoModelForCausalLM.from_pretrained(
            ZeroGPUConfig.MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create pipeline optimized for character generation
        print("🎭 Creating character generation pipeline...")
        weather_model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        # Test with character-like generation
        print("🧪 Testing character generation...")
        test_result = weather_model(
            "You are Napoleon Dynamite. The weather is sunny.",
            max_new_tokens=15,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"✅ Character test successful")
        
        model_loading_status = "loaded"
        print("✅ Character models initialized successfully!")
        return True
        
    except Exception as e:
        model_loading_status = "failed"
        print(f"❌ Model initialization failed: {e}")
        return False

# === Weather API Handler ===
class WeatherAPI:
    def __init__(self):
        self.session = None
    
    @lru_cache(maxsize=ZeroGPUConfig.MAX_CACHE_SIZE)
    def get_weather(self, city: str) -> Dict:
        """Get weather data with caching"""
        try:
            return asyncio.run(self._get_weather_async(city))
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_demo_data(city)
    
    async def _get_weather_async(self, city: str) -> Dict:
        """Async weather fetch"""
        city = self._validate_city(city)
        return self._get_demo_data(city)  # Use demo for now
    
    def _validate_city(self, city: str) -> str:
        if not city or not city.strip():
            return "Ottawa"
        clean_city = city.strip()
        return clean_city if re.match(r"^[a-zA-Z\s\-\.,']{2,}$", clean_city) else "Ottawa"
    
    def _get_demo_data(self, city: str) -> Dict:
        """Character-friendly demo data with variety"""
        # Summer conditions for July (Northern Hemisphere)
        summer_conditions = [
            "Sunny", "Partly Cloudy", "Mostly Sunny", "Clear", 
            "Scattered Clouds", "Light Rain", "Thunderstorms", 
            "Hot and Humid", "Warm", "Perfect Summer Day"
        ]
        
        # July temperatures (realistic for summer)
        base_temp = random.randint(22, 32)  # 22-32°C for summer
        condition = random.choice(summer_conditions)
        
        # Add some dramatic summer weather occasionally
        if random.random() < 0.3:  # 30% chance of dramatic weather
            dramatic_conditions = ["Thunderstorms", "Heavy Rain", "Very Hot", "Heat Wave"]
            condition = random.choice(dramatic_conditions)
            if "Hot" in condition or "Heat" in condition:
                base_temp = random.randint(33, 40)
        
        current_data = {
            'city': city,
            'country': 'Demo',
            'current': {
                'temp_c': base_temp,
                'feels_like_c': base_temp + random.randint(-2, 5),
                'humidity': random.randint(40, 85),
                'wind_kph': random.randint(5, 25),
                'condition': condition,
                'uv': random.randint(6, 11),  # Higher UV for summer
                'description': self._get_weather_description(condition, base_temp)
            }
        }
        
        return current_data
    
    def _get_weather_description(self, condition: str, temp: int) -> str:
        """Get descriptive weather for characters to react to"""
        descriptions = {
            "Sunny": f"bright and sunny with {temp}°C",
            "Thunderstorms": f"dramatic thunderstorms rolling in at {temp}°C",
            "Heavy Rain": f"heavy rain pouring down at {temp}°C", 
            "Very Hot": f"scorching hot at {temp}°C",
            "Heat Wave": f"intense heat wave at {temp}°C",
            "Perfect Summer Day": f"absolutely perfect at {temp}°C"
        }
        return descriptions.get(condition, f"{condition.lower()} at {temp}°C")

# === Character Response Generator ===
class CharacterResponseGenerator:
    def __init__(self):
        self.response_cache = {}
        self.generation_lock = threading.Lock()
    
    def generate_character_response(self, weather_data: Dict, city: str, character: str, 
                                  forecast_range: str, temp_unit: str = "Celsius") -> str:
        """Generate character-specific weather response"""
        
        # Check if model is loaded
        if model_loading_status != "loaded" or weather_model is None:
            return self._create_character_fallback(weather_data, city, character, temp_unit)
        
        # Get character config
        character_config = MOVIE_CHARACTERS.get(character, MOVIE_CHARACTERS["napoleon_dynamite"])
        
        # Create cache key
        cache_key = f"{city}_{character}_{forecast_range}_{temp_unit}_{hash(str(weather_data))}"
        
        # Check cache
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        try:
            # Format weather for character
            weather_context = self._format_weather_for_character(weather_data, forecast_range, temp_unit)
            
            # Create character-specific prompt
            prompt = self._create_character_prompt(weather_context, city, character_config)
            
            # Generate with character personality
            response = self._generate_with_character_gpu(prompt, character_config)
            
            if response and len(response.strip()) > 15:
                self.response_cache[cache_key] = response
                return response
            else:
                return self._create_character_fallback(weather_data, city, character, temp_unit)
            
        except Exception as e:
            print(f"Character generation error: {e}")
            return self._create_character_fallback(weather_data, city, character, temp_unit)
    
    @spaces.GPU
    def _generate_with_character_gpu(self, prompt: str, character_config: Dict) -> str:
        """Generate character response using GPU"""
        with self.generation_lock:
            try:
                if weather_model is None:
                    return ""
                
                result = weather_model(
                    prompt,
                    max_new_tokens=character_config.get('max_tokens', 200),
                    temperature=character_config.get('temperature', 0.9),
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id if tokenizer else None,
                    return_full_text=False
                )
                
                if result and len(result) > 0:
                    response = result[0]["generated_text"]
                    response = self._clean_character_response(response, character_config)
                    return response
                
                return ""
                    
            except Exception as e:
                print(f"GPU character generation error: {e}")
                return ""
    
    def _format_weather_for_character(self, weather_data: Dict, temp_unit: str = "Celsius") -> str:
        """Format weather in an engaging way for characters"""
        def convert_temp(celsius_temp):
            if temp_unit == "Fahrenheit":
                return round(celsius_temp * 9/5 + 32)
            return celsius_temp
        
        unit_symbol = "°F" if temp_unit == "Fahrenheit" else "°C"
        
        current = weather_data.get('current', {})
        condition = current.get('condition', 'Unknown')
        temp = convert_temp(current.get('temp_c', 'Unknown'))
        feels_like = convert_temp(current.get('feels_like_c', temp))
        wind = current.get('wind_kph', 0)
        humidity = current.get('humidity', 50)
        
        context = f"The weather today is {condition} with a temperature of {temp}{unit_symbol}"
        if abs(int(feels_like) - int(temp)) > 3:
            context += f" (feels like {feels_like}{unit_symbol})"
        if wind > 20:
            context += f" with strong winds at {wind} km/h"
        if humidity > 80:
            context += " and quite humid"
        elif humidity < 30:
            context += " and very dry"
            
        return context
    
    def _create_character_prompt(self, weather_context: str, city: str, character_config: Dict) -> str:
        """Create character-specific prompt"""
        
        # Character-specific prompt engineering
        prompt = f"""{character_config['prefix']}

WEATHER CONTEXT: {weather_context} in {city}.

INSTRUCTIONS:
- Stay completely in character as {character_config['name']} from {character_config['movie']}
- Give a current weather report, around 80-120 words.
- Use your distinctive speech patterns and personality
- Reference the weather conditions in your unique style
- Include some of your characteristic phrases naturally

{character_config['name']} says about the weather:"""

        return prompt
    
    def _clean_character_response(self, response: str, character_config: Dict) -> str:
        """Clean and enhance character response"""
        if not response:
            return ""
        
        # Remove prompt remnants
        response = response.strip()
        
        # Remove common AI artifacts
        artifacts_to_remove = [
            "WEATHER CONTEXT:", "INSTRUCTIONS:", "says about the weather:",
            character_config['name'] + " says", "The weather", "Weather report"
        ]
        
        for artifact in artifacts_to_remove:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Ensure character voice is maintained
        if not any(phrase.lower() in response.lower() for phrase in character_config['sample_phrases']):
            # Add a characteristic phrase if missing
            if character_config['name'] == "Napoleon Dynamite" and "gosh" not in response.lower():
                response = response + " Gosh!"
            elif "jack_nicholson" in character_config.get('name', '').lower() and "you can't handle" not in response.lower():
                response = response.replace("weather", "weather... and you can't handle the weather!")
        
        # Ensure proper ending
        if response and not response.endswith(('.', '!', '?')):
            # Character-appropriate endings
            if "napoleon" in character_config.get('name', '').lower():
                response += ". Idiot."
            elif "terminator" in character_config.get('name', '').lower():
                response += ". I'll be back."
            elif "gandalf" in character_config.get('name', '').lower():
                response += "!"
            else:
                response += "."
        
        return response
    
    def _create_character_fallback(self, weather_data: Dict, city: str, character: str, temp_unit: str = "Celsius") -> str:
        """Create character-specific fallback responses"""
        character_config = MOVIE_CHARACTERS.get(character, MOVIE_CHARACTERS["napoleon_dynamite"])
        current = weather_data.get('current', {})
        
        # Convert temperature
        temp_c = current.get('temp_c', 20)
        if temp_unit == "Fahrenheit":
            temp = round(temp_c * 9/5 + 32)
            unit = "°F"
        else:
            temp = temp_c
            unit = "°C"
            
        condition = current.get('condition', 'Unknown')
        
        emoji = random.choice(character_config['emojis'])
        
        # Character-specific fallback responses
        fallbacks = {
            "napoleon_dynamite": f"{emoji} Gosh, the weather in {city} is {condition} and {temp}{unit}. This is like, the worst weather report ever. Whatever, I don't even care. Idiot.",
            
            "jack_nicholson_colonel": f"{emoji} You want the weather report for {city}? YOU CAN'T HANDLE THE WEATHER! It's {condition} at {temp}{unit}, and that's the truth!",
            
            "yoda": f"{emoji} {condition} in {city}, the weather is. {temp}{unit}, the temperature reads. Strong with the Force, this weather forecast is, hmm.",
            
            "tony_stark": f"{emoji} Let me just consult my advanced weather algorithms... {city} is looking at {condition} weather, {temp}{unit}. That's how I roll.",
            
            "forrest_gump": f"{emoji} Well, mama always said weather is like a box of chocolates - you never know what you're gonna get. Today in {city} we got {condition} at {temp}{unit}. That's all I got to say about that.",
            
            "the_dude": f"{emoji} Far out, man. The weather in {city} is like, {condition} at {temp}{unit}, dude. That's just, like, nature's opinion, man. The Dude abides.",
            
            "terminator": f"{emoji} Weather scan complete. {city}: {condition}. Temperature: {temp}{unit}. Mission parameters met. I'll be back... with tomorrow's forecast.",
            
            "gandalf": f"{emoji} The weather in {city} speaks of {condition} at {temp}{unit}! You shall not pass... without knowing the forecast! *dramatic staff gesture*",
            
            "austin_powers": f"{emoji} Yeah baby! The groovy weather in {city} is {condition} at {temp}{unit}! Shagadelic weather reporting, if I do say so myself!",
            
            "deadpool": f"{emoji} Oh great, I'm in a weather app now? *breaks fourth wall* The weather in {city} is {condition} at {temp}{unit}. Maximum effort! This is so stupid it just might work."
        }
        
        return fallbacks.get(character, fallbacks["napoleon_dynamite"])

# === Visualization ===
class WeatherVisualizer:
    @staticmethod
    def create_character_themed_chart(weather_data: Dict, forecast_range: str, character: str) -> Optional[go.Figure]:
        """Create character-themed weather chart"""
        if forecast_range == 'today':
            return WeatherVisualizer.create_character_gauge(weather_data, character)
        
        forecast = weather_data.get('forecast', [])
        if not forecast:
            return None
        
        try:
            dates = [day['date'] for day in forecast]
            highs = [day['max_temp_c'] for day in forecast]
            lows = [day['min_temp_c'] for day in forecast]
            
            # Character-themed colors
            character_colors = {
                "napoleon_dynamite": {"high": "#FF6B6B", "low": "#4ECDC4"},
                "jack_nicholson_colonel": {"high": "#FF0000", "low": "#8B0000"},
                "yoda": {"high": "#00FF00", "low": "#32CD32"},
                "tony_stark": {"high": "#FFD700", "low": "#FF4500"},
                "forrest_gump": {"high": "#87CEEB", "low": "#4682B4"},
                "the_dude": {"high": "#DDA0DD", "low": "#9370DB"},
                "terminator": {"high": "#C0C0C0", "low": "#708090"},
                "gandalf": {"high": "#8A2BE2", "low": "#4B0082"},
                "austin_powers": {"high": "#FF69B4", "low": "#FF1493"},
                "deadpool": {"high": "#DC143C", "low": "#B22222"}
            }
            
            colors = character_colors.get(character, {"high": "#FF6B6B", "low": "#4ECDC4"})
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=highs, name='High Temp',
                line=dict(color=colors["high"], width=3),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=lows, name='Low Temp',
                line=dict(color=colors["low"], width=3),
                mode='lines+markers'
            ))
            
            character_config = MOVIE_CHARACTERS.get(character, MOVIE_CHARACTERS["napoleon_dynamite"])
            
            fig.update_layout(
                title=f'{character_config["name"]} Weather Forecast - {weather_data.get("city", "Unknown")}',
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0.05)'
            )
            
            return fig
            
        except Exception as e:
            print(f"Character visualization error: {e}")
            return None
    
    @staticmethod
    def create_character_gauge(weather_data: Dict, character: str) -> Optional[go.Figure]:
        """Create character-themed temperature gauge"""
        try:
            current = weather_data.get('current', {})
            temp = current.get('temp_c', 20)
            character_config = MOVIE_CHARACTERS.get(character, MOVIE_CHARACTERS["napoleon_dynamite"])
            
            # Character-themed gauge colors
            gauge_colors = {
                "napoleon_dynamite": "orange",
                "jack_nicholson_colonel": "red", 
                "yoda": "green",
                "tony_stark": "gold",
                "forrest_gump": "blue",
                "the_dude": "purple",
                "terminator": "silver",
                "gandalf": "indigo",
                "austin_powers": "hotpink",
                "deadpool": "crimson"
            }
            
            color = gauge_colors.get(character, "orange")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = temp,
                title = {
                    'text': f"{character_config['name']}'s Weather<br>{weather_data.get('city', 'Unknown')}"
                },
                gauge = {
                    'axis': {'range': [-20, 40]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [-20, 0], 'color': "lightblue"},
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "lightcoral"}
                    ]
                }
            ))
            
            fig.update_layout(height=350)
            return fig
            
        except Exception as e:
            print(f"Character gauge error: {e}")
            return None

# === Initialize Components ===
weather_api = WeatherAPI()
character_generator = CharacterResponseGenerator()
visualizer = WeatherVisualizer()

# === Main Chat Function ===
def character_weather_chat(history, city, character, temp_unit, session_id):
    """Character-based weather chat"""
    try:
        # Check if character is selected
        if character is None:
            error_msg = "🎬 Please select a character first!"
            if history is None:
                history = []
            new_history = list(history)
            new_history.append(("Select a character", error_msg))
            return new_history, new_history
        
        # Get weather data
        weather_data = weather_api.get_weather(city)
        
        # Generate character response with temperature unit
        response = character_generator.generate_character_response(
            weather_data, city, character, temp_unit
        )
        
        # Format user query with character
        character_config = MOVIE_CHARACTERS.get(character, MOVIE_CHARACTERS["napoleon_dynamite"])
        user_query = f"🎬 {character_config['name']} - Weather for {city}"
        
        # Update history
        if history is None:
            history = []
        
        new_history = list(history)
        new_history.append((user_query, response))
        
        return new_history, new_history
        
    except Exception as e:
        error_msg = f"🎬 Character error: {str(e)}"
        print(f"Character chat error: {e}")
        
        if history is None:
            history = []
        
        new_history = list(history)
        new_history.append((f"Error: {city}", error_msg))
        return new_history, new_history

# === Character tile selection function ===
def select_character(character_key):
    """Handle character tile selection"""
    return character_key

# === Gradio Interface ===
def create_character_weather_interface():
    """Create movie character weather interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
        title="🎬 Movie Character Weather"
    ) as demo:
        
        # Header
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
            <h1>🎬 Movie Character Weather Reports</h1>
            <p>Get your weather forecast from iconic movie characters!</p>
            <p style="opacity: 0.9;">From Napoleon Dynamite's awkward complaints to Jack Nicholson's dramatic intensity</p>
        </div>
        """)
        
        # Status
        status_display = gr.HTML(f"""
        <div style="background: linear-gradient(90deg, #3F51B5, #2196F3); color: white; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center; font-weight: bold;">
            🎬 Character AI Loading... • {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode'} • Demo Weather Data
        </div>
        """)
        
        # Row 1: City and temperature unit
        with gr.Row():
            city_input = gr.Textbox(
                label="🏙️ City",
                placeholder="Enter city name...",
                value="Ottawa",
                scale=2
            )
            
            temp_unit = gr.Radio(
                choices=["Celsius", "Fahrenheit"],
                value="Celsius",
                label="🌡️ Temperature Unit",
                scale=1
            )
        
        # Row 2: Character selection and get weather button
        with gr.Row():
            with gr.Column():
                # Hidden character selector (controlled by tiles) - no default
                character_select = gr.State(value=None)
                # Display selected character
                selected_character_display = gr.Markdown("**Please select a character**", elem_id="selected-display")
                
                # Character tiles selection
                gr.Markdown("### 🎭 Choose Your Character")
                
                # Character-specific symbolic colors based on their movies/personalities
                tile_colors = {
                    "napoleon_dynamite": "#FF6B35",  # Orange - Idaho/quirky/nostalgic
                    "jack_nicholson_colonel": "#8B0000",  # Military green/red - authority/intensity
                    "yoda": "#228B22",  # Green - Force/wisdom
                    "tony_stark": "#FFD700",  # Gold - wealth/tech/arc reactor
                    "forrest_gump": "#87CEEB",  # Sky blue - innocence/running
                    "the_dude": "#8B4513",  # Brown - bowling/white russian/earthy
                    "terminator": "#C0C0C0",  # Silver/metallic - machine/metal
                    "gandalf": "#9400D3",  # Purple - magic/mysticism
                    "austin_powers": "#FF1493",  # Hot pink - 60s groovy/psychedelic
                    "deadpool": "#DC143C"  # Crimson red - costume/violence/comedy
                }
                
                # Create character tiles in a 5x2 grid (5 columns, 2 rows)
                with gr.Row():
                    for i in range(5):
                        if i < len(MOVIE_CHARACTERS):
                            char_key = list(MOVIE_CHARACTERS.keys())[i]
                            char_config = MOVIE_CHARACTERS[char_key]
                            color = tile_colors.get(char_key, "#667eea")
                            
                            # Create clickable character tile with custom color
                            tile_btn = gr.Button(
                                f"{random.choice(char_config['emojis'])} {char_config['name']}\n{char_config['movie']}",
                                size="sm",
                                elem_classes=f"character-tile character-tile-{char_key}",
                                elem_id=f"tile-{char_key}"
                            )
                            
                            # Update selection and visual feedback
                            def update_selection(char_key=char_key):
                                char_name = MOVIE_CHARACTERS[char_key]['name']
                                return char_key, f"**Selected: {char_name}**"
                            
                            tile_btn.click(
                                fn=update_selection,
                                outputs=[character_select, selected_character_display],
                                queue=False
                            )
                
                with gr.Row():
                    for i in range(5, 10):
                        if i < len(MOVIE_CHARACTERS):
                            char_key = list(MOVIE_CHARACTERS.keys())[i]
                            char_config = MOVIE_CHARACTERS[char_key]
                            color = tile_colors.get(char_key, "#667eea")
                            
                            # Create clickable character tile with custom color
                            tile_btn = gr.Button(
                                f"{random.choice(char_config['emojis'])} {char_config['name']}\n{char_config['movie']}",
                                size="sm",
                                elem_classes=f"character-tile character-tile-{char_key}",
                                elem_id=f"tile-{char_key}"
                            )
                            
                            # Update selection and visual feedback
                            def update_selection(char_key=char_key):
                                char_name = MOVIE_CHARACTERS[char_key]['name']
                                return char_key, f"**Selected: {char_name}**"
                            
                            tile_btn.click(
                                fn=update_selection,
                                outputs=[character_select, selected_character_display],
                                queue=False
                            )
                
                # Get weather button below character selection
                get_weather_btn = gr.Button("🎬 Get Character Weather", variant="primary", size="lg")
        
        # Row 3: Chat display and clear button
        with gr.Row():
            with gr.Column():
                chat_display = gr.Chatbot(
                    label="Character Weather Chat",
                    height=500,
                    show_copy_button=True,
                    type="tuples"
                )
                
                # Clear button below chat
                clear_btn = gr.Button("🧹 Clear Chat", variant="secondary", size="lg")
        

        
        # State
        conversation_state = gr.State([])
        session_state = gr.State(lambda: str(random.randint(100000, 999999)))
        
        # Events
        get_weather_btn.click(
            fn=character_weather_chat,
            inputs=[conversation_state, city_input, character_select, temp_unit, session_state],
            outputs=[chat_display, conversation_state]
        )
        
        clear_btn.click(
            fn=lambda: ([], []),
            outputs=[chat_display, conversation_state]
        )
        
        # Auto-submit
        city_input.submit(
            fn=character_weather_chat,
            inputs=[conversation_state, city_input, character_select, temp_unit, session_state],
            outputs=[chat_display, conversation_state]
        )
        
        # Enhanced startup
        def startup():
            try:
                success = initialize_zerogpu_models()
                
                if success:
                    return f"""
                    <div style="background: linear-gradient(90deg, #00BCD4, #2196F3); color: white; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center; font-weight: bold;">
                        🎬 Character AI Ready! • {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode'} • Demo Weather Data (Summer 2025) ✅
                    </div>
                    """
                else:
                    return f"""
                    <div style="background: linear-gradient(90deg, #FF5722, #F44336); color: white; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center; font-weight: bold;">
                        🎬 Fallback Mode • {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode'} • Using preset character responses & demo weather ⚠️
                    </div>
                    """
                    
            except Exception as e:
                return f"""
                <div style="background: linear-gradient(90deg, #ff6b6b, #ee5a24); color: white; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center; font-weight: bold;">
                    🎬 Character Loading Error • Please refresh ❌
                </div>
                """
        
        demo.load(fn=startup, outputs=[status_display])
        
        # Footer with custom styling
        gr.HTML("""
        <style>
        .character-tile {
            height: 70px !important;
            white-space: pre-line !important;
            font-size: 0.85em !important;
            transition: all 0.3s ease !important;
            border: 3px solid transparent !important;
            font-weight: 500 !important;
            position: relative !important;
            padding: 8px !important;
        }
        .character-tile:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        /* Selected character indicator */
        .character-tile.selected {
            border-color: #2196F3 !important;
            box-shadow: 0 0 20px rgba(33, 150, 243, 0.4) !important;
            transform: scale(1.05) !important;
        }
        .character-tile.selected::after {
            content: "✓";
            position: absolute;
            top: 5px;
            right: 5px;
            background: #2196F3;
            color: white;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.8em;
        }
        
        /* Individual character tile colors - symbolic of their movies/personalities */
        .character-tile-napoleon_dynamite {
            background: linear-gradient(135deg, #FF6B35, #FF8C42) !important;  /* Orange - quirky/nostalgic */
            color: white !important;
        }
        .character-tile-jack_nicholson_colonel {
            background: linear-gradient(135deg, #8B0000, #B22222) !important;  /* Dark red - intensity/anger */
            color: white !important;
        }
        .character-tile-yoda {
            background: linear-gradient(135deg, #228B22, #32CD32) !important;  /* Green - The Force */
            color: white !important;
        }
        .character-tile-tony_stark {
            background: linear-gradient(135deg, #FFD700, #FFA500) !important;  /* Gold - arc reactor/wealth */
            color: #1a1a1a !important;
        }
        .character-tile-forrest_gump {
            background: linear-gradient(135deg, #87CEEB, #4682B4) !important;  /* Sky blue - innocence/running */
            color: white !important;
        }
        .character-tile-the_dude {
            background: linear-gradient(135deg, #8B4513, #A0522D) !important;  /* Brown - bowling/earthy */
            color: white !important;
        }
        .character-tile-terminator {
            background: linear-gradient(135deg, #C0C0C0, #708090) !important;  /* Silver - metallic/machine */
            color: #1a1a1a !important;
        }
        .character-tile-gandalf {
            background: linear-gradient(135deg, #9400D3, #8A2BE2) !important;  /* Purple - magic/mysticism */
            color: white !important;
        }
        .character-tile-austin_powers {
            background: linear-gradient(135deg, #FF1493, #FF69B4) !important;  /* Hot pink - 60s psychedelic */
            color: white !important;
        }
        .character-tile-deadpool {
            background: linear-gradient(135deg, #DC143C, #FF0000) !important;  /* Crimson - costume/blood */
            color: white !important;
        }
        
        /* Selected display styling */
        #selected-display {
            text-align: center;
            font-size: 1.1em;
            color: #2196F3;
            margin: 10px 0;
            padding: 8px;
            background: rgba(33, 150, 243, 0.1);
            border-radius: 8px;
        }
        
        /* Button styling */
        .lg {
            margin-top: 15px !important;
            padding: 12px 24px !important;
        }
        </style>
        
        <script>
        // JavaScript to handle visual selection
        document.addEventListener('DOMContentLoaded', function() {
            // No default selection
            
            // Handle tile clicks
            document.querySelectorAll('.character-tile').forEach(tile => {
                tile.addEventListener('click', function() {
                    // Remove selected class from all tiles
                    document.querySelectorAll('.character-tile').forEach(t => {
                        t.classList.remove('selected');
                    });
                    // Add selected class to clicked tile
                    this.classList.add('selected');
                });
            });
        });
        </script>
        """)
    
    return demo

# === Launch ===
if __name__ == "__main__":
    print("🎬 Starting Movie Character Weather App...")
    demo = create_character_weather_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
