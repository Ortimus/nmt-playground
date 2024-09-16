from transformers import MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer, MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_CONFIGS = {
    'marian': {
        'model_class': MarianMTModel,
        'tokenizer_class': MarianTokenizer,
        'name_format': 'Helsinki-NLP/opus-mt-{}-{}',
        'supported_pairs': [('en', 'de'), ('de', 'en'), ('en', 'fr'), ('fr', 'en'), ('en', 'es'), ('es', 'en')],
        'prefix': ''
    },
    'm2m100': {
        'model_class': M2M100ForConditionalGeneration,
        'tokenizer_class': M2M100Tokenizer,
        'name_format': 'facebook/m2m100_418M',
        'supported_pairs': [('en', 'de'), ('de', 'en'), ('en', 'fr'), ('fr', 'en'), ('en', 'es'), ('es', 'en'), ('en', 'zh'), ('zh', 'en')],
        'prefix': ''
    }
}

""" 
    'mbart50': {
        'model_class': MBartForConditionalGeneration,
        'tokenizer_class': MBart50TokenizerFast,
        'name_format': 'facebook/mbart-large-50-many-to-many-mmt',
        'supported_pairs': [('en', 'de'), ('de', 'en'), ('en', 'fr'), ('fr', 'en'), ('en', 'es'), ('es', 'en'), ('en', 'zh'), ('zh', 'en')],
        'prefix': ''
    },
    'nllb_small': {
        'model_class': AutoModelForSeq2SeqLM,
        'tokenizer_class': AutoTokenizer,
        'name_format': 'facebook/nllb-200-distilled-100M',
        'supported_pairs': [('en', 'de'), ('de', 'en'), ('en', 'fr'), ('fr', 'en'), ('en', 'es'), ('es', 'en'), ('en', 'zh'), ('zh', 'en')],
        'prefix': '{} => {}: '
    } 
"""

def get_model_and_tokenizer(model_type, source_lang, target_lang):
    config = MODEL_CONFIGS[model_type]
    if model_type == 'marian':
        model_name = config['name_format'].format(source_lang, target_lang)
    else:
        model_name = config['name_format']
    model = config['model_class'].from_pretrained(model_name)
    tokenizer = config['tokenizer_class'].from_pretrained(model_name)
    return model, tokenizer

def get_supported_language_pairs(model_type):
    return MODEL_CONFIGS[model_type]['supported_pairs']

def get_available_models():
    return list(MODEL_CONFIGS.keys())

def get_model_prefix(model_type, source_lang, target_lang):
    config = MODEL_CONFIGS[model_type]
    if model_type == 'nllb_small':
        return config['prefix'].format(convert_lang_code(model_type, source_lang), 
                                       convert_lang_code(model_type, target_lang))
    return config['prefix']

def convert_lang_code(model_type, lang_code):
    if model_type == 'nllb_small':
        code_map = {
            'en': 'eng_Latn', 'de': 'deu_Latn', 'fr': 'fra_Latn', 
            'es': 'spa_Latn', 'zh': 'zho_Hans', 'ar': 'ara_Arab',
            'ru': 'rus_Cyrl', 'ja': 'jpn_Jpan', 'ko': 'kor_Hang',
            'hi': 'hin_Deva', 'pt': 'por_Latn', 'it': 'ita_Latn'
        }
        return code_map.get(lang_code, lang_code)
    elif model_type == 'mbart50':
        return f"{lang_code}_XX"
    return lang_code