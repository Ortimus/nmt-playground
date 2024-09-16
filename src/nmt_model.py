import torch.nn as nn
from model_factory import get_model_and_tokenizer, get_model_prefix, convert_lang_code, get_supported_language_pairs
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="clean_up_tokenization_spaces")

class NMT(nn.Module):
    def __init__(self, model_type='marian'):
        super(NMT, self).__init__()
        self.model_type = model_type
        self.models = {}
        self.tokenizers = {}

    def get_model_and_tokenizer(self, source_lang, target_lang):
        lang_pair = (source_lang, target_lang)
        if lang_pair not in self.models:
            model, tokenizer = get_model_and_tokenizer(self.model_type, source_lang, target_lang)
            self.models[lang_pair] = model
            self.tokenizers[lang_pair] = tokenizer
        return self.models[lang_pair], self.tokenizers[lang_pair]

    def translate(self, sentences, source_lang, target_lang, **kwargs):
        model, tokenizer = self.get_model_and_tokenizer(source_lang, target_lang)
        
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512

        if self.model_type == 'mbart50':
            tokenizer.src_lang = convert_lang_code(self.model_type, source_lang)
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=kwargs['max_length'])
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[convert_lang_code(self.model_type, target_lang)], **kwargs)
        elif self.model_type == 'm2m100':
            tokenizer.src_lang = source_lang
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=kwargs['max_length'])
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang), **kwargs)
        elif self.model_type == 'nllb_small':
            prefix = get_model_prefix(self.model_type, source_lang, target_lang)
            prefixed_sentences = [prefix + sent for sent in sentences]
            inputs = tokenizer(prefixed_sentences, return_tensors="pt", padding=True, truncation=True, max_length=kwargs['max_length'])
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[convert_lang_code(self.model_type, target_lang)], **kwargs)
        else:  # marian
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=kwargs['max_length'])
            translated = model.generate(**inputs, **kwargs)
        
        return tokenizer.batch_decode(translated, skip_special_tokens=True)

    def get_supported_language_pairs(self):
        return get_supported_language_pairs(self.model_type)

    def forward(self, source, target, source_lang, target_lang):
        model, tokenizer = self.get_model_and_tokenizer(source_lang, target_lang)
        
        inputs = tokenizer(source, return_tensors="pt", padding=True, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target, return_tensors="pt", padding=True, truncation=True).input_ids

        outputs = model(**inputs, labels=labels)
        return outputs.loss

    def compute_bleu_score(self, reference, hypothesis):
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        return bleu.corpus_score([hypothesis], [[reference]]).score