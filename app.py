import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect

# Initialize conversation history and user details in Streamlit session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Load the required models with caching for performance
@st.cache_resource
def load_models():
    medqa_model = pipeline("text2text-generation", model="emilykang/Phi_medmcqa_question_generation-gynaecology_n_obstetrics_lora")
    instruct_model = pipeline("text-generation", model="microsoft/Phi-3.5-MoE-instruct")
    return medqa_model, instruct_model

# Multilingual translation with caching
@st.cache_data
def translate_text(text, source_lang, target_lang='en'):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text

# Detect keywords related to gynecology and obstetrics
def is_gyn_related(text):
    gyn_keywords = ['pregnancy', 'gynecology', 'obstetrics', 'birth', 'fertility', 'contraception', 'menstruation']
    return any(keyword in text.lower() for keyword in gyn_keywords)

# Automatically detect the input language
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        st.error(f"Language detection error: {e}")
        return 'en'  # Default to English if detection fails

def main():
    st.title("Gynecology AI Chatbot with Doctor Reminder")
    
    # Ask for user's name if not already provided
    if not st.session_state.user_name:
        st.session_state.user_name = st.text_input("What is your name?", "")
    
    # Load models
    medqa_model, instruct_model = load_models()

    # Input from the user and detected language
    user_input = st.text_area(f"Your question, {st.session_state.user_name}:", "", height=150)

    if user_input.strip():
        # Detect language automatically
        detected_lang = detect_language(user_input)

        # Translate input to English
        translated_input = translate_text(user_input, source_lang=detected_lang)

        # Store the user's question in the conversation history
        st.session_state.conversation_history.append(f"{st.session_state.user_name}: {translated_input}")

        # Combine the conversation history for model input
        conversation_input = " ".join(st.session_state.conversation_history)

        # Generate response based on conversation history with loading spinner
        with st.spinner('Generating response...'):
            if is_gyn_related(translated_input):
                response = medqa_model(conversation_input)[0]['generated_text']
            else:
                response = instruct_model(conversation_input)[0]['generated_text']

        # Append doctor reminder to the response
        response += "\n\n*Please remember to consult with a healthcare professional for accurate diagnosis and advice.*"

        # Store the AI's response in the conversation history
        st.session_state.conversation_history.append(f"AI: {response}")

        # Translate response back to the user's detected language
        translated_response = translate_text(response, source_lang='en', target_lang=detected_lang)

        # Display the response
        st.subheader(f"Response (Detected language: {detected_lang}):")
        st.write(translated_response)

        # Display the conversation history
        st.subheader("Conversation History")
        for entry in st.session_state.conversation_history:
            st.write(entry)
    else:
        st.write("Please enter a question to get started.")

if __name__ == "__main__":
    main()
