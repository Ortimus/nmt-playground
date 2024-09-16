import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from nmt_model import NMT
from model_factory import get_available_models

class TestNMT(unittest.TestCase):
    """
    Comprehensive test suite for the Neural Machine Translation (NMT) class.

    This class contains unit tests to verify the functionality of the NMT class,
    including model initialization, translation, and beam search
    for multiple language pairs and model types.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment before running any tests.

        This method is called once before any tests in this class are run.
        It initializes NMT models for each available model type and sets up sample sentences for different language pairs.
        """
        cls.models = {model_type: NMT(model_type) for model_type in get_available_models()}
        cls.test_pairs = {
            ('en', 'de'): {
                'source': ["Hello, how are you?", "The weather is nice today.", "I love programming."],
                'target': ["Hallo, wie geht es dir?", "Das Wetter ist heute schön.", "Ich liebe Programmieren."]
            },
            ('en', 'fr'): {
                'source': ["Hello, how are you?", "The weather is nice today.", "I love programming."],
                'target': ["Bonjour, comment allez-vous ?", "Le temps est beau aujourd'hui.", "J'adore programmer."]
            },
            ('en', 'es'): {
                'source': ["Hello, how are you?", "The weather is nice today.", "I love programming."],
                'target': ["Hola, ¿cómo estás?", "El tiempo es agradable hoy.", "Me encanta programar."]
            }
        }

    def test_initialization(self):
        """
        Test the proper initialization of the NMT models.

        This test checks if the models are instances of NMT class.
        """
        for model_type, model in self.models.items():
            with self.subTest(model_type=model_type):
                self.assertIsInstance(model, NMT)

    def test_supported_language_pairs(self):
        """
        Test that each model type supports the expected language pairs.
        """
        for model_type, model in self.models.items():
            with self.subTest(model_type=model_type):
                supported_pairs = model.get_supported_language_pairs()
                self.assertIsInstance(supported_pairs, list)
                self.assertGreater(len(supported_pairs), 0)

    def test_translate(self):
        """
        Test the translation functionality of the NMT models for multiple language pairs.

        This test checks if the translate method produces valid translations
        for given input sentences in different language pairs.
        """
        for model_type, model in self.models.items():
            for (source_lang, target_lang), data in self.test_pairs.items():
                if (source_lang, target_lang) in model.get_supported_language_pairs():
                    with self.subTest(model_type=model_type, source_lang=source_lang, target_lang=target_lang):
                        translations = model.translate(data['source'], source_lang, target_lang)
                        self.assertEqual(len(translations), len(data['source']))
                        for translation in translations:
                            self.assertIsInstance(translation, str)
                            self.assertGreater(len(translation), 0)

    def test_beam_search(self):
        """
        Test the beam search functionality of the NMT models.

        This test checks if the beam search method produces the expected number
        of hypotheses with the correct structure for a given input sentence in different language pairs.
        """
        beam_size = 3
        for model_type, model in self.models.items():
            for (source_lang, target_lang), data in self.test_pairs.items():
                if (source_lang, target_lang) in model.get_supported_language_pairs():
                    with self.subTest(model_type=model_type, source_lang=source_lang, target_lang=target_lang):
                        hypotheses = model.translate(
                            [data['source'][0]], 
                            source_lang, 
                            target_lang, 
                            num_beams=beam_size, 
                            num_return_sequences=beam_size
                        )
                        self.assertEqual(len(hypotheses), beam_size)
                        for hyp in hypotheses:
                            self.assertIsInstance(hyp, str)
                            self.assertGreater(len(hyp), 0)

if __name__ == '__main__':
    unittest.main()