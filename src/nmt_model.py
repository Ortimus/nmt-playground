import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

class NMT(nn.Module):
    """
    Neural Machine Translation class using pre-trained MarianMT models.
    
    This class provides a high-level interface for neural machine translation
    using Hugging Face's implementations of MarianMT models. It supports
    translation between specified language pairs.

    Limitations:
    - Only supports language pairs available in the MODEL_MAPPING.
    - Relies on pre-trained models, which may not perform well on domain-specific texts.
    - No fine-tuning mechanism is currently implemented.
    """

    MODEL_MAPPING = {
        ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
        ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
        # Add more language pairs as needed
    }

    def __init__(self, source_lang, target_lang):
        """
        Initialize the NMT model for a specific language pair.

        Args:
            source_lang (str): Source language code (e.g., 'en' for English)
            target_lang (str): Target language code (e.g., 'de' for German)

        Raises:
            ValueError: If the language pair is not supported.

        Note:
            This method loads a pre-trained model and tokenizer for the specified language pair.
            It does not support custom training or fine-tuning in its current implementation.
        """
        super(NMT, self).__init__()
        self.source_lang = source_lang
        self.target_lang = target_lang
        model_name = self.MODEL_MAPPING.get((source_lang, target_lang))
        if not model_name:
            raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)

    def forward(self, source, target):
        """
        Perform a forward pass through the model.

        This method is used for training and computing loss. It tokenizes the input,
        passes it through the model, and returns the loss.

        Args:
            source (List[str]): List of source sentences
            target (List[str]): List of target sentences

        Returns:
            torch.Tensor: The loss value

        Note:
            This method is primarily used for training. For translation, use the
            beam_search method instead.
        """
        source_ids = self.tokenizer(source, return_tensors="pt", padding=True, truncation=True).input_ids
        target_ids = self.tokenizer(target, return_tensors="pt", padding=True, truncation=True).input_ids
        
        outputs = self.model(input_ids=source_ids, labels=target_ids)
        
        return outputs.loss

    def encode(self, source_padded, source_lengths):
        """
        Encode the source sentences.

        This method runs the encoder part of the model on the input sentences.

        Args:
            source_padded (torch.Tensor): Padded source sentences
            source_lengths (List[int]): Original lengths of source sentences

        Returns:
            Tuple[torch.Tensor, Tuple[None, None]]: 
                - Encoder's last hidden state
                - Tuple of (None, None) for compatibility with some interfaces

        Note:
            The source_lengths parameter is not used in the current implementation
            but is kept for interface compatibility.
        """
        encoder_outputs = self.model.get_encoder()(source_padded)
        return encoder_outputs.last_hidden_state, (None, None)

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded):
        """
        Decode the target sentences given the encoder hidden states.

        This method runs the decoder part of the model.

        Args:
            enc_hiddens (torch.Tensor): Encoder hidden states
            enc_masks (torch.Tensor): Encoder masks (not used in current implementation)
            dec_init_state (Tuple[torch.Tensor, torch.Tensor]): Initial decoder state (not used)
            target_padded (torch.Tensor): Padded target sentences

        Returns:
            torch.Tensor: Decoder's last hidden state

        Note:
            enc_masks and dec_init_state are not used in the current implementation
            but are kept for interface compatibility.
        """
        decoder_outputs = self.model.get_decoder()(
            input_ids=target_padded,
            encoder_hidden_states=enc_hiddens
        )
        return decoder_outputs.last_hidden_state

    def beam_search(self, src_sent, beam_size=5, max_decoding_time_step=70):
        """
        Perform beam search decoding for translation.

        This method translates a single source sentence using beam search decoding.

        Args:
            src_sent (str): Source sentence to translate
            beam_size (int): Size of the beam for beam search
            max_decoding_time_step (int): Maximum number of decoding steps

        Returns:
            List[Dict]: List of hypotheses, each containing 'value' (translated tokens) and 'score'

        Note:
            This method uses the model's generate method, which implements beam search internally.
            The 'score' in the output is currently not populated and is set to 0.0.
        """
        inputs = self.tokenizer([src_sent], return_tensors="pt", padding=True, truncation=True)
        
        translations = self.model.generate(
            **inputs,
            num_beams=beam_size,
            max_length=max_decoding_time_step,
            early_stopping=True,
            num_return_sequences=beam_size
        )
        
        decoded_translations = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
        
        hypotheses = [{"value": trans.split(), "score": 0.0} for trans in decoded_translations]
        return hypotheses