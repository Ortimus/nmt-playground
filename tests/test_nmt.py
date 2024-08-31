import unittest
import torch
from src.nmt_model import NMT

class TestNMT(unittest.TestCase):
    """
    Comprehensive test suite for the Neural Machine Translation (NMT) class.

    This class contains unit tests to verify the functionality of the NMT class,
    including model initialization, encoding, decoding, forward pass, beam search,
    and translation for multiple language pairs.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment before running any tests.

        This method is called once before any tests in this class are run.
        It initializes an NMT model and sets up sample sentences for different language pairs.
        """
        cls.model = NMT()
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
        Test the proper initialization of the NMT model.

        This test checks if the model is an instance of NMT class.
        """
        self.assertIsInstance(self.model, NMT)

    def test_load_language_pair(self):
        """
        Test the loading of language pair models.

        This test verifies if language pair models are loaded correctly for supported pairs
        and if an error is raised for unsupported pairs.
        """
        for lang_pair in self.test_pairs.keys():
            self.model.load_language_pair(*lang_pair)
            self.assertIn(lang_pair, self.model.models)
            self.assertIn(lang_pair, self.model.tokenizers)

        with self.assertRaises(ValueError):
            self.model.load_language_pair('en', 'ja')  # Assuming English to Japanese is not supported

    def test_encode(self):
        """
        Test the encoding functionality of the NMT model.

        This test verifies if the encode method produces the expected output shape
        and type for a given input in different language pairs.
        """
        for (source_lang, target_lang), data in self.test_pairs.items():
            self.model.load_language_pair(source_lang, target_lang)
            lang_pair = (source_lang, target_lang)
            source_ids = self.model.tokenizers[lang_pair](data['source'], return_tensors="pt", padding=True, truncation=True).input_ids
            encoder_outputs = self.model.models[lang_pair].get_encoder()(source_ids)
            self.assertIsInstance(encoder_outputs.last_hidden_state, torch.Tensor)
            self.assertEqual(encoder_outputs.last_hidden_state.shape[0], len(data['source']))

    def test_decode(self):
        """
        Test the decoding functionality of the NMT model.

        This test checks if the decode method produces output of the expected
        shape and type given encoder hidden states and target sentences for different language pairs.
        """
        for (source_lang, target_lang), data in self.test_pairs.items():
            self.model.load_language_pair(source_lang, target_lang)
            lang_pair = (source_lang, target_lang)
            source_ids = self.model.tokenizers[lang_pair](data['source'], return_tensors="pt", padding=True, truncation=True).input_ids
            target_ids = self.model.tokenizers[lang_pair](data['target'], return_tensors="pt", padding=True, truncation=True).input_ids
            encoder_outputs = self.model.models[lang_pair].get_encoder()(source_ids)
            decoder_outputs = self.model.models[lang_pair].get_decoder()(
                input_ids=target_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state
            )
            self.assertIsInstance(decoder_outputs.last_hidden_state, torch.Tensor)
            self.assertEqual(decoder_outputs.last_hidden_state.shape[0], len(data['target']))

    def test_forward(self):
        """
        Test the forward pass of the NMT model for multiple language pairs.

        This test verifies if the forward method computes a valid loss
        for given source and target sentences in different language pairs.
        """
        for (source_lang, target_lang), data in self.test_pairs.items():
            loss = self.model(data['source'], data['target'], source_lang, target_lang)
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size([]))  # scalar loss
            self.assertGreater(loss.item(), 0)  # loss should be positive

    def test_translate(self):
        """
        Test the translation functionality of the NMT model for multiple language pairs.

        This test checks if the translate method produces valid translations
        for given input sentences in different language pairs.
        """
        for (source_lang, target_lang), data in self.test_pairs.items():
            translations = self.model.translate(data['source'], source_lang, target_lang)
            self.assertEqual(len(translations), len(data['source']))
            for translation in translations:
                self.assertIsInstance(translation, str)
                self.assertGreater(len(translation), 0)

    def test_beam_search(self):
        """
        Test the beam search functionality of the NMT model.

        This test checks if the beam search method produces the expected number
        of hypotheses with the correct structure for a given input sentence in different language pairs.
        """
        beam_size = 3
        for (source_lang, target_lang), data in self.test_pairs.items():
            hypotheses = self.model.translate(
                [data['source'][0]], 
                source_lang, 
                target_lang, 
                beam_size=beam_size, 
                num_return_sequences=beam_size
            )
            self.assertEqual(len(hypotheses), 1)  # One list of hypotheses for one input sentence
            self.assertEqual(len(hypotheses[0]), beam_size)  # beam_size number of hypotheses
            for hyp in hypotheses[0]:
                self.assertIsInstance(hyp, str)
                self.assertGreater(len(hyp), 0)


if __name__ == '__main__':
    unittest.main()