# Neural Machine Translation (NMT) Project

This project implements a flexible Neural Machine Translation system using pre-trained models from Hugging Face's Transformers library. It supports multiple language pairs and provides an easy-to-use interface for translation tasks.

## Features

- Support for multiple language pairs
- Utilizes pre-trained MarianMT models
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


## Acknowledgments

This project uses models from the Hugging Face Transformers library.