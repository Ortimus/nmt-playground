# Neural Machine Translation (NMT) Project

This project implements a flexible Neural Machine Translation system using pre-trained models from Hugging Face's Transformers library. It supports multiple language pairs and provides an easy-to-use interface for translation tasks.

## Features

- Support for multiple language pairs
- Utilizes pre-trained MarianMT models (https://huggingface.co/docs/transformers/en/model_doc/marian)
- Beam search for improved translation quality
- Easy-to-use API for translation tasks
- Comprehensive test suite

## Installation

1. Clone the repository:
```
  git clone https://github.com/Ortimus/nmt-project.git
  cd nmt-project
```

2. Create a virtual environment:
```
python -m venv nmt_env
source nmt_env/bin/activate 
```
3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the NMT model:

```python
from src.nmt_model import NMT

# Initialize the model
nmt = NMT()

# Translate a sentence from English to German
source_sentence = "Hello, how are you?"
translated = nmt.translate([source_sentence], source_lang='en', target_lang='de')
print(translated[0])
```

## Supported Language Pairs
Currently, the following language pairs are supported:

- English to German (en-de)
- German to English (de-en)
- English to French (en-fr)
- French to English (fr-en)
- English to Spanish (en-es)
- Spanish to English (es-en)

More language pairs can be added by updating the MODEL_MAPPING in the NMT class.

## Running Tests
To run the test suite:

```
python -m unittest discover tests
```

## Note on Beam Search
This project uses beam search for translation. Beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. It's used to balance the needs of better translation quality and efficient computation.

## Limitations

Relies on pre-trained models, which may not perform well on domain-specific texts.
No fine-tuning mechanism is currently implemented.
Performance may vary depending on the specific language pair and the domain of the text.



## Streamlit Application

This project includes a Streamlit web application that provides a user-friendly interface for the Neural Machine Translation model.

### Features

- Interactive web interface for translation
- Support for multiple language pairs
- Adjustable translation parameters (beam size, max length, number of translations)
- Real-time translation results

### Running the Streamlit App

To run the Streamlit application, follow these steps:

1. Ensure you have activated your virtual environment:
2. Navigate to the project directory:
```
cd path/to/nmt-project
```
4. Run the Streamlit app:
```
streamlit run src/app.py
```
5. The app should now be running. Open your web browser and go to the URL displayed in your terminal (typically `http://localhost:8501`).


### Using the App

1. Select the source and target languages from the dropdown menus in the sidebar.

2. Adjust the translation parameters if desired:
   - Beam Size: Controls the breadth of the search during translation. Higher values may yield better results but increase computation time.
   - Max Length: Sets the maximum length of the generated translation.
   - Number of Translations: Determines how many translation variants to generate. This value will automatically adjust to be at most the Beam Size.

3. Enter the text you want to translate in the "Source Text" input box.

4. (Optional) Enter a reference translation in the "Reference Translation" input box. This allows the system to compute BLEU scores, which measure translation quality.

5. Click the "Translate" button to see the results.

6. The results will be displayed in a table format:
   - If no reference translation was provided, you'll see the generated translations.
   - If a reference translation was provided, you'll also see BLEU scores for each translation, indicating how close they are to the reference.

Note: The "Number of Translations" will always be less than or equal to the "Beam Size". If you increase the Beam Size, you'll have the option to generate more translation variants.

### Customization

You can customize the app's appearance by modifying the `.streamlit/config.toml` file in the project root directory. This file allows you to change colors, fonts, and other visual elements of the Streamlit app.

### Troubleshooting

If you encounter any issues:
- Ensure all required packages are installed (`pip install -r requirements.txt`)
- Check that you're using the correct Python environment
- Make sure you're running the app from the project root directory

For more information on Streamlit, visit [Streamlit's documentation](https://docs.streamlit.io/).

## Acknowledgments

This project uses models from the Hugging Face Transformers library.