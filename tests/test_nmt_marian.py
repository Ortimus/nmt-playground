from transformers import MarianMTModel, MarianTokenizer

def translate(text, model_name='Helsinki-NLP/opus-mt-en-de'):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return result[0]

if __name__ == "__main__":
    text = "Hello, how are you?"
    result = translate(text)
    print(f"Original: {text}")
    print(f"Translated: {result}")