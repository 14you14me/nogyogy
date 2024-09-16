import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer from Hugging Face using transformers
@st.cache_resource
def load_model():
    # Load the model and tokenizer for microsoft/Phi-3.5-MoE-instruct
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
    return model, tokenizer

# Function to generate text based on user input
def generate_response(model, tokenizer, prompt, max_length=200):
    # Tokenize the input prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI for interaction
def main():
    st.title("Gynecology AI Chatbot")

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # User input for the chatbot
    user_input = st.text_area("Ask your question about gynecology or obstetrics:", "", height=150)

    # If the user provides input, generate a response
    if user_input:
        with st.spinner("Generating response..."):
            response = generate_response(model, tokenizer, user_input)
        st.write(response)

if __name__ == "__main__":
    main()
