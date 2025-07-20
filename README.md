# 🎬 Movie Character Weather App

Get your weather forecast from iconic movie characters! From Napoleon Dynamite's awkward complaints to Gandalf's mystical predictions, experience weather reports like never before.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **10 Iconic Movie Characters**: Each with unique personality and speech patterns
- **AI-Powered Responses**: Uses DialoGPT for character-specific weather reports
- **Temperature Units**: Toggle between Celsius and Fahrenheit
- **Character-Themed UI**: Each character has symbolic colors from their movies
- **ZeroGPU Optimized**: Efficient GPU usage on HuggingFace Spaces
- **Interactive Design**: Beautiful, responsive interface built with Gradio

## 🎭 Featured Characters

| Character | Movie | Personality |
|-----------|-------|-------------|
| Napoleon Dynamite | Napoleon Dynamite (2004) | Awkward, deadpan teenager |
| Colonel Jessup | A Few Good Men (1992) | Intense military officer |
| Yoda | Star Wars Saga | Wise Jedi Master |
| Tony Stark | Marvel Cinematic Universe | Genius billionaire playboy |
| Forrest Gump | Forrest Gump (1994) | Simple, sincere soul |
| The Dude | The Big Lebowski (1998) | Laid-back philosopher |
| The Terminator | Terminator Series | Robotic precision |
| Gandalf | Lord of the Rings | Wise wizard |
| Austin Powers | Austin Powers Series | Groovy 60s spy |
| Deadpool | Deadpool Movies | Fourth-wall breaking antihero |

## 🚀 Live Demo

Try it out on HuggingFace Spaces: [Movie Character Weather App](https://huggingface.co/spaces/YOUR_USERNAME/movie-character-weather)

## 💻 Technical Stack

- **Frontend**: Gradio 4.0+
- **AI Model**: Microsoft DialoGPT-small
- **Framework**: Transformers by HuggingFace
- **Deployment**: HuggingFace Spaces with ZeroGPU
- **Language**: Python 3.8+

## 🛠️ Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/movie-character-weather.git
cd movie-character-weather
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python app.py
```

5. Open your browser to `http://localhost:7860`

### Requirements

```txt
gradio>=4.0.0
transformers>=4.30.0
torch>=2.0.0
spaces
plotly>=5.0.0
aiohttp
requests
```

## 🏗️ Architecture

```
movie-character-weather/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Git ignore file
```

### Key Components

1. **Character System**
   - Each character has unique personality traits
   - Custom prompts for authentic responses
   - Character-specific temperature and token settings

2. **Weather Generation**
   - Demo weather data (can be connected to real APIs)
   - Summer-appropriate conditions for current date
   - Dynamic weather descriptions

3. **AI Response Generation**
   - DialoGPT model for natural language generation
   - Character-specific prompt engineering
   - Response caching for performance

4. **UI/UX Design**
   - Responsive Gradio interface
   - Character tiles with symbolic colors
   - Clean, intuitive layout

## 🎨 Character Color Symbolism

- **Napoleon Dynamite** (Orange): Quirky 2000s nostalgia
- **Colonel Jessup** (Dark Red): Military intensity
- **Yoda** (Green): The Force and wisdom
- **Tony Stark** (Gold): Wealth and technology
- **Forrest Gump** (Sky Blue): Innocence and freedom
- **The Dude** (Brown): Earthy, laid-back vibes
- **Terminator** (Silver): Metallic machine aesthetic
- **Gandalf** (Purple): Magic and mysticism
- **Austin Powers** (Hot Pink): 60s psychedelic
- **Deadpool** (Crimson): His iconic costume

## 🔧 Configuration

### Environment Variables (Optional)

```bash
# For real weather data (currently using demo data)
OPENWEATHER_API_KEY=your_api_key_here
WEATHERAPI_KEY=your_api_key_here
```

### Model Configuration

The app uses `microsoft/DialoGPT-small` for efficiency. You can modify the model in `ZeroGPUConfig`:

```python
MODEL_NAME = "microsoft/DialoGPT-small"  # Can be changed to medium/large
MAX_RESPONSE_TOKENS = 250
TEMPERATURE = 0.9
```

## 📝 Usage

1. **Select a City**: Enter any city name
2. **Choose Temperature Unit**: Celsius or Fahrenheit
3. **Pick a Character**: Click on any character tile
4. **Get Weather**: Click the button to receive your character's weather report
5. **Enjoy**: Read the weather in your favorite character's voice!

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute:

- Add new movie characters
- Improve character responses
- Connect real weather APIs
- Enhance UI/UX design
- Fix bugs or improve performance

### Adding a New Character

1. Add character config to `MOVIE_CHARACTERS` dictionary:
```python
"character_key": {
    "name": "Character Name",
    "movie": "Movie Title (Year)",
    "personality": "Brief description",
    "speech_style": "How they talk",
    "sample_phrases": ["Phrase 1", "Phrase 2"],
    "prefix": "Character prompt prefix",
    "emojis": ["🎭", "🎬"],
    "temperature": 0.9,
    "max_tokens": 200
}
```

2. Add character color in the UI section
3. Create fallback response in `_create_character_fallback()`

## 🐛 Known Issues

- Some characters may occasionally break character in complex scenarios
- GPU availability on HuggingFace Spaces may affect response time

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for Spaces and Transformers
- Microsoft for DialoGPT
- Gradio for the amazing UI framework
- All the movies that gave us these iconic characters

## 📞 Contact

- **GitHub**: [YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- **HuggingFace**: [Your HF Profile](https://huggingface.co/YOUR_USERNAME)

---

<p align="center">Made with ❤️ and 🎬 by [Your Name]</p>

<p align="center">
  <i>"Gosh! This weather app is like, totally sweet!" - Napoleon Dynamite</i>
</p>
