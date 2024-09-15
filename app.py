import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from langdetect import detect
from datetime import date

# Initialize session states for conversation, user details, and logged symptoms
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

if 'symptom_log' not in st.session_state:
    st.session_state.symptom_log = []

# Load smaller models with caching and CPU optimization
@st.cache_resource
def load_models():
    # Use smaller models like distilgpt2 for text generation
    instruct_model = pipeline("text-generation", model="distilgpt2", device=-1)
    # Use t5-small as a lightweight text2text model
    medqa_model = pipeline("text2text-generation", model="t5-small", device=-1)
    return medqa_model, instruct_model

# Translation functions with caching to minimize repeated calls
@st.cache_data
def translate_text(text, source_lang, target_lang='en'):
    if source_lang != target_lang:
        try:
            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        except Exception as e:
            st.error(f"Translation Error: {e}")
            return text
    return text  # If the source and target languages are the same, avoid unnecessary translation.

# Detect keywords related to gynecology and specific conditions
def detect_condition(text):
    conditions = {
        "pregnancy": ["pregnancy", "pregnant", "missed period", "nausea"],
        "fertility": ["fertility", "ovulation", "infertility", "trying to conceive"],
        "menstruation": ["period", "menstruation", "irregular periods", "cramps"],
        "contraception": ["contraception", "birth control", "pills", "condoms"],
    }

    for condition, keywords in conditions.items():
        if any(keyword in text.lower() for keyword in keywords):
            return condition
    return None

# Follow-up questions based on the condition detected
def follow_up_questions(condition):
    questions = {
        "pregnancy": "Are you experiencing any unusual pain or spotting?",
        "fertility": "How long have you been trying to conceive?",
        "menstruation": "Are your periods usually irregular?",
        "contraception": "Are you using any birth control methods currently?",
    }
    return questions.get(condition, None)

# Doctor reminder with localization
def doctor_reminder(detected_lang):
    reminder = "Please remember to consult with a healthcare professional for accurate diagnosis and advice."
    return translate_text(reminder, source_lang="en", target_lang=detected_lang)

# Enhance response based on condition and detected language
def enhance_response(response, user_name, detected_lang, condition):
    tips = ""

    if condition == "pregnancy":
        tips = ("Here are some general tips during pregnancy:\n"
                "- Stay hydrated\n"
                "- Eat a balanced diet\n"
                "- Avoid smoking and alcohol\n"
                "- Schedule regular prenatal check-ups\n")
    elif condition == "fertility":
        tips = ("To improve fertility:\n"
                "- Maintain a healthy weight\n"
                "- Track ovulation cycles\n"
                "- Consider fertility tests if you've been trying to conceive for over a year\n")
    elif condition == "menstruation":
        tips = ("Tips for menstrual health:\n"
                "- Track your cycle to notice any irregularities\n"
                "- Maintain a balanced diet\n"
                "- Exercise regularly to reduce cramps\n")
    elif condition == "contraception":
        tips = ("General birth control tips:\n"
                "- Take birth control pills at the same time each day\n"
                "- Use condoms to protect against STIs\n"
                "- Consider long-term methods like IUDs or implants if needed\n")

    if tips:
        response += f"\n\n**Additional Tips for You, {user_name}:**\n{tips}"

    response += f"\n\n*{doctor_reminder(detected_lang)}*"
    return response

# Log symptoms for the user
def log_symptoms(symptom, log_date):
    st.session_state.symptom_log.append({"date": log_date, "symptom": symptom})

# Display symptom log history
def display_symptom_log():
    if st.session_state.symptom_log:
        st.write("### Symptom Log History")
        for entry in st.session_state.symptom_log:
            st.write(f"- {entry['date']}: {entry['symptom']}")
    else:
        st.write("No symptoms have been logged yet.")

def main():
    st.title("Optimized Gynecology AI Chatbot for CPU")

    # Ask for user's name if not already provided
    if not st.session_state.user_name:
        st.session_state.user_name = st.text_input("What is your name?", "")

    # Load models
    medqa_model, instruct_model = load_models()

    # Input from the user and detected language
    user_input = st.text_area(f"Your question, {st.session_state.user_name}:", "", height=150)

    # Optional symptom logging input
    symptom = st.text_input("Log a symptom (optional):")
    log_date = st.date_input("Date", value=date.today())
    if symptom:
        log_symptoms(symptom, log_date)

    display_symptom_log()

    if user_input.strip():
        # Detect language automatically
        detected_lang = detect(user_input)  # Use langdetect library to detect input language

        # Translate input to English
        translated_input = translate_text(user_input, source_lang=detected_lang)

        # Store the user's question in the conversation history
        st.session_state.conversation_history.append(f"{st.session_state.user_name}: {translated_input}")

        # Detect condition based on input
        detected_condition = detect_condition(translated_input)

        # Combine the conversation history for model input
        conversation_input = " ".join(st.session_state.conversation_history)

        # Generate response based on conversation history with loading spinner
        with st.spinner('Generating response...'):
            if detected_condition:
                response = medqa_model(conversation_input)[0]['generated_text']
            else:
                response = instruct_model(conversation_input)[0]['generated_text']

        # Enhance the response with tips, condition-based advice, and doctor reminder
        enhanced_response = enhance_response(response, st.session_state.user_name, detected_lang, detected_condition)

        # Store the AI's response in the conversation history
        st.session_state.conversation_history.append(f"AI: {enhanced_response}")

        # Translate response back to the user's detected language
        translated_response = translate_text(enhanced_response, source_lang='en', target_lang=detected_lang)

        # Display the response
        st.subheader(f"Response (Detected language: {detected_lang}):")
        st.write(translated_response)

        # Display the conversation history
        st.subheader("Conversation History")
        for entry in st.session_state.conversation_history:
            st.write(entry)

        # If a condition was detected, ask a follow-up question
        if detected_condition:
            follow_up = follow_up_questions(detected_condition)
            if follow_up:
                st.subheader("Follow-Up Question")
                st.write(follow_up)
    else:
        st.write("Please enter a question to get started.")

if __name__ == "__main__":
    main()
