import streamlit as st
import torch
import pandas as pd
from nmt_model import NMT
from model_factory import get_available_models


@st.cache_resource
def load_model(model_type):
    return NMT(model_type)

def main():
    st.set_page_config(layout="wide")
    st.title("Neural Machine Translation")

    # Model selection
    available_models = get_available_models()
    model_type = st.sidebar.selectbox("Select Model", available_models, key="model_select")
    model = load_model(model_type)

    # Language pair selection
    supported_pairs = model.get_supported_language_pairs()
    source_languages = list(set([pair[0] for pair in supported_pairs]))
    default_source = 'en' if 'en' in source_languages else source_languages[0]
    source_lang = st.sidebar.selectbox("Source Language", 
                                       source_languages, 
                                       index=source_languages.index(default_source),
                                       key="source_lang_select")
    
    target_languages = [pair[1] for pair in supported_pairs if pair[0] == source_lang]
    target_lang = st.sidebar.selectbox("Target Language", 
                                       target_languages,
                                       key="target_lang_select")

    # Translation parameters
    st.sidebar.header("Translation Parameters")
    beam_size = st.sidebar.slider("Beam Size", 
                                  min_value=1, 
                                  max_value=10, 
                                  value=5, 
                                  help="Number of beams for beam search. Higher values may give better results but increase computation time.",
                                  key="beam_size_slider")
    
    max_length = st.sidebar.slider("Max Length", 
                                   min_value=10, 
                                   max_value=512,  # Changed from 500 to 512
                                   value=100, 
                                   step=10,
                                   help="Maximum length of the generated translation.",
                                   key="max_length_slider")
    
    num_return_sequences = st.sidebar.slider("Number of Translations", 
                                            min_value=1, 
                                            max_value=max(5, beam_size),
                                            value=min(3, beam_size),
                                            help="Number of translations to return. These correspond to the predictions with the highest probabilities.",
                                            key="num_return_sequences_slider")

    
    # Add instructions to the sidebar
    with st.sidebar.expander("How to Use"):
        st.markdown("""
        1. Select source and target languages from the dropdowns above.

        2. Adjust translation parameters:
           - Beam Size: Controls search breadth. Higher values may improve results but increase computation time.
           - Max Length: Sets the maximum length of the generated translation.
           - Number of Translations: Determines translation variants to generate. Will not exceed Beam Size.

        3. Enter text to translate in the "Source Text" box in the main area.

        4. (Optional) Enter a reference translation for BLEU score computation.

        5. Click "Translate" to see results.

        6. Results will show in a table:
           - Without reference: Generated translations.
           - With reference: Translations and BLEU scores.

        Note: "Number of Translations" is always ≤ "Beam Size".
        """)
    
    # Input text
    st.header(f"Enter text to translate ({source_lang})")
    source_text = st.text_area("Source Text", height=150, key="source_text_area")

    # Reference translation
    st.header(f"Enter reference translation (optional, for BLEU score) ({target_lang})")
    reference_text = st.text_area("Reference Translation", height=150, key="reference_text_area")

    if st.button("Translate", key="translate_button"):
        if source_text:
            with st.spinner('Translating...'):
                try:
                    translations = model.translate(
                        [source_text], 
                        source_lang, 
                        target_lang, 
                        num_beams=beam_size,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences
                    )
                    
                    # Ensure translations is a list
                    if not isinstance(translations, list):
                        translations = [translations]
                    elif len(translations) == 1 and isinstance(translations[0], list):
                        translations = translations[0]
                    
                    # Prepare data for the table
                    table_data = []
                    for _, translation in enumerate(translations, 1):
                        row = {"Translation": translation}
                        if reference_text:
                            bleu_score = model.compute_bleu_score(reference_text, translation)
                            row["BLEU Score"] = f"{bleu_score:.2f}"
                        table_data.append(row)
                    
                    # Create DataFrame
                    df = pd.DataFrame(table_data)
                    
                    # Display results
                    st.header(f"Translation Results ({target_lang})")
                    st.table(df)
                    
                    st.success("Translation completed successfully!")
                    
                except Exception as e:
                    st.error(f"An error occurred during translation: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")


    # Display Model Information 
    with st.sidebar.expander("Model Information"):
        st.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        st.write("Supported Language Pairs:")
        for pair in supported_pairs:
            st.write(f"- {pair[0]} to {pair[1]}")

    st.markdown("---")
    st.markdown("Built using Streamlit and Hugging Face Transformers")




def main_0():

    st.set_page_config(layout="wide")
    st.title("Neural Machine Translation")

    # Model selection
    available_models = get_available_models()
    model_type = st.sidebar.selectbox("Select Model", available_models)
    model = load_model(model_type)

    # Language pair selection
    supported_pairs = model.get_supported_language_pairs()
    source_languages = list(set([pair[0] for pair in supported_pairs]))
    default_source = 'en' if 'en' in source_languages else source_languages[0]
    source_lang = st.sidebar.selectbox("Source Language", source_languages, index=source_languages.index(default_source))
    target_languages = [pair[1] for pair in supported_pairs if pair[0] == source_lang]
    target_lang = st.sidebar.selectbox("Target Language", target_languages)
    
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
                                            max_value=max(5, beam_size),  # Ensure max is not greater than beam_size
                                            value=min(3, beam_size), # Ensure default is not greater than beam_size
                                            help="Number of translations to return. These correspond to the predictions with  the highest probablities.")

    
    # Add instructions to the sidebar
    with st.sidebar.expander("How to Use"):
        st.markdown("""
        1. Select source and target languages from the dropdowns above.

        2. Adjust translation parameters:
           - Beam Size: Controls search breadth. Higher values may improve results but increase computation time.
           - Max Length: Sets the maximum length of the generated translation.
           - Number of Translations: Determines translation variants to generate. Will not exceed Beam Size.

        3. Enter text to translate in the "Source Text" box in the main area.

        4. (Optional) Enter a reference translation for BLEU score computation.

        5. Click "Translate" to see results.

        6. Results will show in a table:
           - Without reference: Generated translations.
           - With reference: Translations and BLEU scores.

        Note: "Number of Translations" is always ≤ "Beam Size".
        """)
        
    # Input text
    st.header(f"Enter text to translate ({source_lang})")
    source_text = st.text_area("Source Text", height=150)
    

    # Add a text area for reference translation
    st.header(f"Enter reference translation (optional, for BLEU score) ({target_lang})")
    reference_text = st.text_area("Reference Translation", height=150)

    if st.button("Translate"):
        if source_text:
            with st.spinner('Translating...'):
                try:
                    translations = model.translate(
                        [source_text], 
                        source_lang, 
                        target_lang, 
                        num_beams=beam_size,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences
                    )
                    
                    # Ensure translations is a list
                    if not isinstance(translations, list):
                        translations = [translations]
                    elif len(translations) == 1 and isinstance(translations[0], list):
                        translations = translations[0]
                    
                    # Prepare data for the table
                    table_data = []
                    for i, translation in enumerate(translations, 1):
                        row = {"Translation": translation}
                        if reference_text:
                            bleu_score = model.compute_bleu_score(reference_text, translation)
                            row["BLEU Score"] = f"{bleu_score:.2f}"
                        table_data.append(row)
                    
                    # Create DataFrame
                    df = pd.DataFrame(table_data)
                    
                    # Display results
                    st.header(f"Translation Results ({target_lang})")
                    st.table(df)
                    
                    st.success("Translation completed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during translation: {str(e)}")
        else:
            st.warning("Please enter some text to translate.")
                    


    # Display Model Information 
    with st.sidebar.expander("Model Information"):
        st.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        st.write("Supported Language Pairs:")
        for pair in supported_pairs:
            st.write(f"- {pair[0]} to {pair[1]}")

    st.markdown("---")
    st.markdown("Built using Streamlit and Hugging Face Transformers")

if __name__ == "__main__":
    main()