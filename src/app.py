import streamlit as st
import torch
from nmt_model import NMT

@st.cache_resource
def load_model():
    return NMT()

def main():

    st.title("Neural Machine Translation")

    # Load the model
    model = load_model()

    # Language pair selection
    st.sidebar.header("Language Settings")
    supported_pairs = model.get_supported_language_pairs()
    
    # Get all unique source languages
    source_languages = list(set([pair[0] for pair in supported_pairs]))
    
    # Set 'en' as default if it exists in source languages
    default_source = 'en' if 'en' in source_languages else source_languages[0]
    
    source_lang = st.sidebar.selectbox("Source Language", source_languages, index=source_languages.index(default_source))
    
    # Filter target languages based on selected source language
    target_languages = [pair[1] for pair in supported_pairs if pair[0] == source_lang]
    target_lang = st.sidebar.selectbox("Target Language", target_languages)

    # Translation parameters
    st.sidebar.header("Translation Parameters")
    beam_size = st.sidebar.slider("Beam Size", 
                                  min_value=1, 
                                  max_value=10, 
                                  value=5, 
                                  help="Number of beams for beam search. Higher values may give better results but increase computation time.")
    
    max_length = st.sidebar.slider("Max Length", 
                                   min_value=10, 
                                   max_value=500, 
                                   value=100, 
                                   step=10,
                                   help="Maximum length of the generated translation.")
    
    num_return_sequences = st.sidebar.slider("Number of Translations", 
                                             min_value=1, 
                                             max_value=5,
                                            value=1,
                                            help="Number of translations to return. These correspond to the predictions with  the highest probablities.")

    # Input text
    st.header(f"Enter text to translate ({source_lang})")
    source_text = st.text_area("Source Text", height=150)

    if st.button("Translate"):
        if source_text:
            with st.spinner('Translating...'):
                try:
                    # Perform translation
                    translations = model.translate(
                        [source_text], 
                        source_lang=source_lang, 
                        target_lang=target_lang,
                        beam_size=beam_size,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences
                    )

                    # Display translations
                    st.header(f"Translation Results ({target_lang})")
                    if num_return_sequences > 1:
                        for i, translation in enumerate(translations[0], 1):
                            st.write(f"Translation {i}: {translation}")
                    else:
                        st.write(f"Translation: {translations[0]}")

                    if translations:
                        st.success("Translation completed successfully!")

                except ValueError as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")

    # Display model information
    with st.expander("Model Information"):
        st.sidebar.header("Model Information")
        st.sidebar.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        st.sidebar.write("Supported Language Pairs:")
        for pair in supported_pairs:
            st.sidebar.write(f"- {pair[0]} to {pair[1]}")

    st.markdown("---")
    st.markdown("Built using Streamlit and Hugging Face Transformers")

if __name__ == "__main__":
    main()