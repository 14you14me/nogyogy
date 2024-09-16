import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to load the model and tokenizer with 8-bit quantization
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("emilykang/Phi_medner-obstetrics_gynecology")
        
        # Load the model with 8-bit quantization using bitsandbytes
        model = AutoModelForCausalLM.from_pretrained(
            "emilykang/Phi_medner-obstetrics_gynecology", 
            load_in_8bit=True,  # Apply 8-bit quantization
            device_map="auto",   # Automatically place layers on available devices
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to generate text from the model
def generate_text(model, tokenizer, prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main Streamlit app
def main():
    st.title("Gynecology AI Chatbot")

    # Load the model and tokenizer
    model, tokenizer = load_model()

    if model is None or tokenizer is None:
        st.error("Failed to load the model.")
        return

    # User input for the chatbot
    user_input = st.text_area("Ask your question about gynecology or obstetrics:", "", height=150)

    # If the user provides input, generate a response
    if user_input:
        with st.spinner("Generating response..."):
            response = generate_text(model, tokenizer, user_input)
        st.write(response)

if __name__ == "__main__":
    main()
