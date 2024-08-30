import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

class NMT(nn.Module):
    """
    Neural Machine Translation class supporting multiple language pairs using pre-trained MarianMT models.
    
    This class provides a high-level interface for neural machine translation
    using Hugging Face's implementations of MarianMT models. It supports
    translation between multiple specified language pairs.

    Attributes:
        MODEL_MAPPING (dict): A mapping of language pairs to their corresponding pre-trained model names.

    Limitations:
    - Only supports language pairs available in the MODEL_MAPPING.
    - Relies on pre-trained models, which may not perform well on domain-specific texts.
    - No fine-tuning mechanism is currently implemented.
    """

    MODEL_MAPPING = {
        ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
        ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
        ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
        ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
        ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
        ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
        # Add more language pairs as needed
    }

    def __init__(self):
        """
        Initialize the NMT model.

        This constructor initializes the base nn.Module and prepares the model to handle multiple language pairs.
        Specific language pair models are loaded on-demand to save memory.
        """
        super(NMT, self).__init__()
        self.models = {}
        self.tokenizers = {}

    def load_language_pair(self, source_lang, target_lang):
        """
        Load a specific language pair model and tokenizer.

        Args:
            source_lang (str): Source language code (e.g., 'en' for English)
            target_lang (str): Target language code (e.g., 'de' for German)

        Raises:
            ValueError: If the language pair is not supported.
        """
        lang_pair = (source_lang, target_lang)
        if lang_pair not in self.MODEL_MAPPING:
            raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")
        
        if lang_pair not in self.models:
            model_name = self.MODEL_MAPPING[lang_pair]
            self.models[lang_pair] = MarianMTModel.from_pretrained(model_name)
            self.tokenizers[lang_pair] = MarianTokenizer.from_pretrained(model_name)

    def forward(self, source, target, source_lang, target_lang):
        """
        Perform a forward pass through the model for a specific language pair.

        This method is used for training and computing loss. It tokenizes the input,
        passes it through the model, and returns the loss.

        Args:
            source (List[str]): List of source sentences
            target (List[str]): List of target sentences
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            torch.Tensor: The loss value

        Note:
            This method is primarily used for training. For translation, use the
            translate method instead.
        """
        lang_pair = (source_lang, target_lang)
        self.load_language_pair(source_lang, target_lang)
        
        source_ids = self.tokenizers[lang_pair](source, return_tensors="pt", padding=True, truncation=True).input_ids
        target_ids = self.tokenizers[lang_pair](target, return_tensors="pt", padding=True, truncation=True).input_ids
        
        outputs = self.models[lang_pair](input_ids=source_ids, labels=target_ids)
        
        return outputs.loss

    def translate(self, sentences, source_lang, target_lang, beam_size=5, max_length=100, num_return_sequences=1):
        """
        Translate sentences from source language to target language.

        Args:
            sentences (List[str]): List of sentences to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            beam_size (int): Size of the beam for beam search
            max_length (int): Maximum length of the generated translation
            num_return_sequences (int): Number of translation sequences to return

        Returns:
            List[str]: List of translated sentences. If num_return_sequences > 1,
                    returns a list of lists, where each inner list contains
                    the multiple translations for a single input sentence.

        Note:
            This method uses the model's generate method, which implements beam search internally.
        """
        lang_pair = (source_lang, target_lang)
        self.load_language_pair(source_lang, target_lang)
        
        inputs = self.tokenizers[lang_pair](sentences, return_tensors="pt", padding=True, truncation=True)
        
        translated = self.models[lang_pair].generate(
            **inputs,
            num_beams=beam_size,
            max_length=max_length,
            early_stopping=True,
            num_return_sequences=num_return_sequences
        )
        
        outputs = self.tokenizers[lang_pair].batch_decode(translated, skip_special_tokens=True)
        
        if num_return_sequences > 1:
            # Reshape the output to group multiple translations for each input sentence
            return [outputs[i:i+num_return_sequences] for i in range(0, len(outputs), num_return_sequences)]
        else:
            return outputs