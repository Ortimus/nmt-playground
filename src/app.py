import streamlit as st
import torch
import pandas as pd
from nmt_model import NMT
from model_factory import get_available_models

# Dictionary to map language codes to full names
LANGUAGE_NAMES = {
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'es': 'Spanish',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'hi': 'Hindi',
    'pt': 'Portuguese',
    'it': 'Italian'
}

@st.cache_resource
def load_model(model_type):
    return NMT(model_type)

def get_language_name(code):
    return LANGUAGE_NAMES.get(code, code)

def main():
    st.set_page_config(layout="wide")
    st.title("Neural Machine Translation")

    # Initialize session state variables
    if 'prev_target_lang' not in st.session_state:
        st.session_state.prev_target_lang = None
    if 'reference_text' not in st.session_state:
        st.session_state.reference_text = ""

    # Model selection
    available_models = get_available_models()
    model_type = st.sidebar.selectbox("Select Model", available_models, key="model_select")
    model = load_model(model_type)

    # Language pair selection
    supported_pairs = model.get_supported_language_pairs()
    
    # Get all unique source languages
    source_languages = list(set([pair[0] for pair in supported_pairs]))
    default_source = 'en' if 'en' in source_languages else source_languages[0]
    
    # Create a dictionary of language codes to full names for the dropdown
    source_language_names = {get_language_name(lang): lang for lang in source_languages}
    
    # Select source language
    source_lang_name = st.sidebar.selectbox(
        "Source Language", 
        options=list(source_language_names.keys()), 
        index=list(source_language_names.keys()).index(get_language_name(default_source)),
        key="source_lang_select"
    )
    source_lang = source_language_names[source_lang_name]

    # Get target languages based on selected source language
    target_languages = [pair[1] for pair in supported_pairs if pair[0] == source_lang]
    target_language_names = {get_language_name(lang): lang for lang in target_languages}

    # If there's a previously selected target language, try to keep it
    if st.session_state.prev_target_lang in target_languages:
        default_target = st.session_state.prev_target_lang
    else:
        default_target = target_languages[0]
        # Clear reference text if target language changed
        st.session_state.reference_text = ""

    # Select target language
    target_lang_name = st.sidebar.selectbox(
        "Target Language", 
        options=list(target_language_names.keys()),
        index=list(target_language_names.keys()).index(get_language_name(default_target)),
        key="target_lang_select"
    )
    target_lang = target_language_names[target_lang_name]

    # Check if target language has changed
    if st.session_state.prev_target_lang != target_lang:
        st.session_state.reference_text = ""  # Clear reference text
    
    # Store the selected target language for next time
    st.session_state.prev_target_lang = target_lang

    # Check if the selected language pair is supported by the current model
    if (source_lang, target_lang) not in supported_pairs:
        st.sidebar.warning(f"The selected language pair ({source_lang_name} to {target_lang_name}) is not supported by the current model. Please choose a different pair.")

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
                                            help="Number of translations to return. These correspond to the predictions with the highest probabilities.")

    
    # Add instructions to the sidebar
    with st.sidebar.expander("How to Use"):
        st.markdown("""
        1. Select a translation model from the dropdown menu.
        2. Choose source and target languages from the dropdowns.
        - Note: Available language pairs may vary depending on the selected model.
        3. Adjust translation parameters:
        - Beam Size: Controls search breadth. Higher values may improve results but increase computation time.
        - Max Length: Sets the maximum length of the generated translation.
        - Number of Translations: Determines how many translation variants to generate.
        4. Enter the text you want to translate in the "Source Text" box in the main area.
        5. (Optional) Enter a reference translation in the target language for BLEU score computation.
        - Note: This will be cleared if you change the target language or select a model that doesn't support the current language pair.
        6. Click "Translate" to generate the translation(s).
        7. Results will be displayed in a table:
        - Without reference: Only the generated translations will be shown.
        - With reference: Both translations and their corresponding BLEU scores will be displayed.
        
        Additional Notes:
        - If you change the model or target language, make sure the desired language pair is supported.
        - The "Number of Translations" cannot exceed the "Beam Size".
        - BLEU scores provide a measure of translation quality, with higher scores indicating better matches to the reference.
        """)

    # Input text
    st.header(f"Enter text to translate ({source_lang_name})")
    source_text = st.text_area("Source Text", height=150)

    # Reference text for BLEU score
    st.header(f"Enter reference translation (optional, for BLEU score) ({target_lang_name})")
    reference_text = st.text_area("Reference Translation", height=150, value=st.session_state.reference_text)
    
    # Update the stored reference text
    st.session_state.reference_text = reference_text

    if st.button("Translate"):
        if source_text:
            if (source_lang, target_lang) in supported_pairs:
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
                        st.header(f"Translation Results ({target_lang_name})")
                        st.table(df)
                        
                        st.success("Translation completed successfully!")

                    except Exception as e:
                        st.error(f"An error occurred during translation: {str(e)}")
            else:
                st.error(f"The selected language pair ({source_lang_name} to {target_lang_name}) is not supported by the current model. Please choose a different pair.")
        else:
            st.warning("Please enter some text to translate.")

    # Display model information
    with st.sidebar.expander("Model Information"):
        st.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        st.write("Supported Language Pairs:")
        for pair in supported_pairs:
            st.write(f"- {get_language_name(pair[0])} to {get_language_name(pair[1])}")

    st.markdown("---")
    st.markdown("Built using Streamlit and Hugging Face Transformers")

if __name__ == "__main__":
    main()