import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_translator import GoogleTranslator
from langdetect import detect

# Predefined responses for common gynecology-related questions
predefined_responses = {
    "cloves pregnancy": "There is limited research on the effects of consuming cloves during pregnancy. It’s generally considered safe in small amounts used in food. However, consult with your healthcare provider before using cloves in medicinal amounts or supplements while pregnant.",
    "pregnancy": "If you think you might be pregnant, common early signs include a missed period, nausea, breast tenderness, fatigue, and frequent urination. To confirm, take a home pregnancy test or consult your healthcare provider."
}

# Function to detect multiple keywords in the user's question
def detect_keywords(text):
    text_lower = text.lower()
    for key, response in predefined_responses.items():
        if all(keyword in text_lower for keyword in key.split()):
            return key
    return None

# Function to check if the question is relevant to gynecology
def is_relevant_to_gynecology(text):
    gynecology_keywords = ["pregnancy", "fertility", "period", "contraception", "gynecology", "pregnant", "birth control", "obstetrics", "miscarriage", "baby", "menstruation"]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in gynecology_keywords)

# Function to translate text to English
def translate_to_english(text, source_lang='auto'):
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception as e:
        st.error(f"Translation to English failed: {e}")
        return text

# Function to translate text back to the original language
def translate_to_original(text, target_lang='auto'):
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        st.error(f"Translation to original language failed: {e}")
        return text

# Function to generate text using a medical model (BioGPT or similar)
def generate_medical_answer(model, tokenizer, prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the medical language model (e.g., BioGPT from Hugging Face)
@st.cache_resource
def load_medical_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
        model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading medical model: {e}")
        return None, None

# Main Streamlit app
def main():
    st.title("Gynecology AI Chatbot")

    # Load the medical model and tokenizer
    model, tokenizer = load_medical_model()

    if model is None or tokenizer is None:
        st.error("Failed to load the medical model.")
        return

    # User input for the chatbot
    user_input = st.text_area("Ask your question about gynecology or obstetrics:", "", height=150)

    # If the user provides input, detect the language and translate if needed
    if user_input:
        # Detect the input language automatically using langdetect
        detected_lang = detect(user_input)
        st.write(f"Detected language: {detected_lang}")

        # Translate input to English
        translated_input = translate_to_english(user_input, source_lang=detected_lang)

        # Check if the question is related to gynecology
        if not is_relevant_to_gynecology(translated_input):
            st.write("It seems like your question is not related to gynecology or obstetrics. Please ask a question related to these topics.")
            return

        # Check for multiple keywords in the translated question
        keyword = detect_keywords(translated_input)
        if keyword:
            # Provide predefined structured response
            response = predefined_responses[keyword]
        else:
            # If no predefined answer, use the medical language model for general medical questions
            with st.spinner("Generating a medical response for your question..."):
                response = generate_medical_answer(model, tokenizer, translated_input)

        # Translate response back to the original language
        translated_response = translate_to_original(response, target_lang=detected_lang)

        # Display the response
        st.write(translated_response)

if __name__ == "__main__":
    main()
