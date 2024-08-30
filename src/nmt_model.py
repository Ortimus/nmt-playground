import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

class NMT(nn.Module):
    MODEL_MAPPING = {
        ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
        ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
        # Add more language pairs as needed
    }

    def __init__(self, source_lang, target_lang):
        super(NMT, self).__init__()  # Call the parent class initializer
        self.source_lang = source_lang
        self.target_lang = target_lang
        model_name = self.MODEL_MAPPING.get((source_lang, target_lang))
        if not model_name:
            raise ValueError(f"Unsupported language pair: {source_lang} to {target_lang}")
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)

    def forward(self, source, target):
        # Tokenize inputs
        source_ids = self.tokenizer(source, return_tensors="pt", padding=True, truncation=True).input_ids
        target_ids = self.tokenizer(target, return_tensors="pt", padding=True, truncation=True).input_ids
        
        # Run the model
        outputs = self.model(input_ids=source_ids, labels=target_ids)
        
        # Return loss
        return outputs.loss

    def encode(self, source_padded, source_lengths):
        encoder_outputs = self.model.get_encoder()(source_padded)
        return encoder_outputs.last_hidden_state, (None, None)

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded):
        decoder_outputs = self.model.get_decoder()(
            input_ids=target_padded,
            encoder_hidden_states=enc_hiddens
        )
        return decoder_outputs.last_hidden_state

    def beam_search(self, src_sent, beam_size=5, max_decoding_time_step=70):
        # Tokenize input
        inputs = self.tokenizer([src_sent], return_tensors="pt", padding=True, truncation=True)
        
        # Generate translations
        translations = self.model.generate(
            **inputs,
            num_beams=beam_size,
            max_length=max_decoding_time_step,
            early_stopping=True,
            num_return_sequences=beam_size
        )
        
        # Decode translations
        decoded_translations = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
        
        # Format output to match original interface
        hypotheses = [{"value": trans.split(), "score": 0.0} for trans in decoded_translations]
        return hypotheses