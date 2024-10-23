import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Force the model to run on CPU
device = torch.device("cpu")
model.to(device)

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("AI Chatbot")
st.write("Ask anything about Artificial Intelligence!")

# Keep track of conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        # Tokenize input and append conversation history
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([torch.tensor(st.session_state.history), new_input_ids], dim=-1) if st.session_state.history else new_input_ids

        # Generate a response
        chat_history_ids = model.generate(bot_input_ids.to(device), max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode the response
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Display bot response and update conversation history
        st.write(f"AI: {bot_response}")
        st.session_state.history = chat_history_ids
        
