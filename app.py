import streamlit as st
import requests

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-MoE-instruct"
HF_API_KEY = "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"  # Replace with your actual Hugging Face API key

# Set up headers for Hugging Face API
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Function to generate text from Hugging Face Inference API
def generate_text(prompt):
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        st.error(f"Error: Hugging Face API returned status {response.status_code}")
        return "Error: Unable to generate response."

# Streamlit UI
def main():
    st.title("Gynecology AI Chatbot")

    # User Input
    user_input = st.text_area("Ask your question about gynecology or obstetrics:", "", height=150)

    # If user input is provided, generate a response
    if user_input:
        with st.spinner("Generating response..."):
            response = generate_text(user_input)
        st.write(response)

if __name__ == "__main__":
    main()
