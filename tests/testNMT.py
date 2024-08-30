import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import unittest
import torch
from src.nmt_model import NMT

class TestNMT(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model = NMT('en', 'de')  # English to German
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
        self.assertIsInstance(self.model, NMT)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.tokenizer)


    def test_encode(self):
        source_ids = self.model.tokenizer(self.source_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
        encoder_outputs, _ = self.model.encode(source_ids, [len(s.split()) for s in self.source_sentences])
        self.assertIsInstance(encoder_outputs, torch.Tensor)
        self.assertEqual(encoder_outputs.shape[0], len(self.source_sentences))

    def test_decode(self):
        source_ids = self.model.tokenizer(self.source_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
        target_ids = self.model.tokenizer(self.target_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
        enc_hiddens, _ = self.model.encode(source_ids, [len(s.split()) for s in self.source_sentences])
        decoder_outputs = self.model.decode(enc_hiddens, None, (None, None), target_ids)
        self.assertIsInstance(decoder_outputs, torch.Tensor)
        self.assertEqual(decoder_outputs.shape[0], len(self.target_sentences))

    def test_forward(self):
        loss = self.model(self.source_sentences, self.target_sentences)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # scalar loss
        self.assertGreater(loss.item(), 0)  # loss should be positive

    def test_beam_search(self):
        hypotheses = self.model.beam_search(self.source_sentences[0], beam_size=3, max_decoding_time_step=50)
        self.assertEqual(len(hypotheses), 3)
        for hyp in hypotheses:
            self.assertIn('value', hyp)
            self.assertIn('score', hyp)
            self.assertIsInstance(hyp['value'], list)
            self.assertIsInstance(hyp['score'], float)

if __name__ == '__main__':
    unittest.main()