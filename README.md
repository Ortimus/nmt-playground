# NMT Playground: Streamlit app to play with Open-Source LLMs

Hey there! Welcome to my Neural Machine Translation (NMT) playground. This project is all about having fun with Streamlit and open-source Large Language Models (LLMs) for translation tasks. 

## What's this all about?

I created this project to experiment with:
- Building interactive web apps using Streamlit
- Playing around with various open-source LLMs for translation
- Seeing how different models perform on translation tasks

It's a sandbox for learning and exploration, not a production-ready tool. So expect some rough edges and have fun experimenting!

## Cool Features

- Try out different open-source translation models 
  - Currently only a couple of light models are supported to allow it to work in Streamlit
- Interactive web interface powered by Streamlit
- Adjust translation parameters and see how they affect results
- Compare translations from different models side-by-side

## Models I'm Playing With

- MarianMT
- M2M100
- Future: 
   - MBart50
   - NLLB (the smaller, distilled version)

Each model has its own quirks and strengths. Try them out and see which one you like best!

## Taking it for a Spin

1. Clone this repo:
```
  git clone https://github.com/Ortimus/nmt-playground.git
  cd nmt-playground
```
2. Set up a virtual environment (optional, but recommended):
```
python -m venv nmt_env
source nmt_env/bin/activate  # On Windows use nmt_env\Scripts\activate
```
3. Install the goodies:
```
pip install -r requirements.txt
```
4. Fire up the Streamlit app:
```
streamlit run src/app.py
```
5. The app should now be running. Open your web browser typically at http://localhost:8501

## Using the App

- Select a translation model from the dropdown menu.
- Choose source and target languages from the dropdowns. Note: Available language pairs may vary depending on the selected model.
- Adjust translation parameters (Beam Size, Max Length, Number of Translations).
- Enter the text you want to translate in the "Source Text" box.
- (Optional) Enter a reference translation for BLEU score computation.
- Click "Translate" to generate the translation(s).
- View results in the displayed table.

## Using the code 
Here's a basic example of how to use the NMT model:

```
from src.nmt_model import NMT

# Initialize the model (default is MarianMT)
nmt = NMT()

# Or choose a specific model type
# nmt = NMT('m2m100')

# Translate a sentence from English to German
source_sentence = "Hello, how are you?"
translated = nmt.translate([source_sentence], source_lang='en', target_lang='de')
print(translated[0])
```

## Running Tests
To run the test suite:
```
python -m unittest discover tests
```

### Note on Beam Search
This project uses beam search for translation. Beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. It's used to balance the needs of better translation quality and efficient computation.

## Limitations
- This is a playground, so don't expect perfect translations!
- No fine-tuning implemented (yet) - maybe a fun future project?
- Performance varies wildly depending on the model and language pair
- Future idea: 
   - Add more models
   - Compare translations from multiple moedls on its oown tab
   - Add cpability to call paid models through their APIs (not on Streamlit)

## License

This project is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Acknowledgments

- For more information on Streamlit, visit [Streamlit's documentation](https://docs.streamlit.io/).
- For Huggingfcae models , visit [Huggingface's documentation](https://huggingface.co/models?other=LLM).

