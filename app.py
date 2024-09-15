import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Load the base model (microsoft/Phi-3.5-MoE-instruct)
@st.cache_resource
def load_base_model():
    # Load the main instruct model from Hugging Face
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
    return base_model, tokenizer

# Load the LoRA model for question generation (Phi_medmcqa)
@st.cache_resource
def load_lora_model():
    # Load the configuration for the LoRA model and the base model required
    config = PeftConfig.from_pretrained("emilykang/Phi_medmcqa_question_generation-gynaecology_n_obstetrics_lora")
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")  # Base model for LoRA
    model = PeftModel.from_pretrained(base_model, "emilykang/Phi_medmcqa_question_generation-gynaecology_n_obstetrics_lora")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    return model, tokenizer

# Function to generate text from the loaded model
def generate_response(model, tokenizer, prompt, max_length=200):
    # Tokenize input and generate response
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function for the Streamlit app
def main():
    st.title("Gynecology AI Chatbot")

    # Dropdown for model selection (choose between the base instruct model or the LoRA model)
    model_choice = st.selectbox("Choose model:", ["Phi-3.5-MoE-instruct", "Phi_medmcqa_question_generation (LoRA)"])

    # Load models based on user choice
    if model_choice == "Phi-3.5-MoE-instruct":
        model, tokenizer = load_base_model()
    else:
        model, tokenizer = load_lora_model()

    # Input from the user
    user_input = st.text_area("Ask your question about gynecology or obstetrics:", "", height=150)

    if user_input:
        # Generate response based on the user's input
        with st.spinner("Generating response..."):
            response = generate_response(model, tokenizer, user_input)
        st.write(response)

if __name__ == "__main__":
    main()
