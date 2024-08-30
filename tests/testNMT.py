import unittest
import torch
from src.nmt_model import NMT

class TestNMT(unittest.TestCase):
    """
    Test suite for the Neural Machine Translation (NMT) class.

    This class contains unit tests to verify the functionality of the NMT class,
    including model initialization, encoding, decoding, forward pass, and beam search.
    It focuses on English to German translation as a test case.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment before running any tests.

        This method is called once before any tests in this class are run.
        It initializes an NMT model for English to German translation and
        sets up sample source and target sentences for testing.
        """
        cls.model = NMT('en', 'de')  # Initialize English to German translation model
        cls.source_sentences = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love programming."
        ]
        cls.target_sentences = [
            "Hallo, wie geht es dir?",
            "Das Wetter ist heute schön.",
            "Ich liebe Programmieren."
        ]

    def test_initialization(self):
        """
        Test the proper initialization of the NMT model.

        This test checks if the model is an instance of NMT class and
        if the model and tokenizer are properly initialized.
        """
        self.assertIsInstance(self.model, NMT)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.tokenizer)

    def test_encode(self):
        """
        Test the encoding functionality of the NMT model.

        This test verifies if the encode method produces the expected output shape
        and type for a given input.
        """
        source_ids = self.model.tokenizer(self.source_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
        encoder_outputs, _ = self.model.encode(source_ids, [len(s.split()) for s in self.source_sentences])
        self.assertIsInstance(encoder_outputs, torch.Tensor)
        self.assertEqual(encoder_outputs.shape[0], len(self.source_sentences))

    def test_decode(self):
        """
        Test the decoding functionality of the NMT model.

        This test checks if the decode method produces output of the expected
        shape and type given encoder hidden states and target sentences.
        """
        source_ids = self.model.tokenizer(self.source_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
        target_ids = self.model.tokenizer(self.target_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
        enc_hiddens, _ = self.model.encode(source_ids, [len(s.split()) for s in self.source_sentences])
        decoder_outputs = self.model.decode(enc_hiddens, None, (None, None), target_ids)
        self.assertIsInstance(decoder_outputs, torch.Tensor)
        self.assertEqual(decoder_outputs.shape[0], len(self.target_sentences))

    def test_forward(self):
        """
        Test the forward pass of the NMT model.

        This test verifies if the forward method computes a valid loss
        for given source and target sentences.
        """
        loss = self.model(self.source_sentences, self.target_sentences)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # scalar loss
        self.assertGreater(loss.item(), 0)  # loss should be positive

    def test_beam_search(self):
        """
        Test the beam search functionality of the NMT model.

        This test checks if the beam search method produces the expected number
        of hypotheses with the correct structure for a given input sentence.
        """
        hypotheses = self.model.beam_search(self.source_sentences[0], beam_size=3, max_decoding_time_step=50)
        self.assertEqual(len(hypotheses), 3)
        for hyp in hypotheses:
            self.assertIn('value', hyp)
            self.assertIn('score', hyp)
            self.assertIsInstance(hyp['value'], list)
            self.assertIsInstance(hyp['score'], float)

if __name__ == '__main__':
    unittest.main()