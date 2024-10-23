import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Force the model to run on CPU
device = torch.device("cpu")
model.to(device)

# Streamlit app
st.title("AI Chatbot")
user_input = st.text_input("You: ")

if user_input:
    # Encode the input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    
    # Generate a response
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the response
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    st.text(f"Chatbot: {response}")
