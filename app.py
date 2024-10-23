import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model and tokenizer
model_name = "fine_tuned_chatbot"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use CPU (or GPU if available)
device = torch.device("cpu")
model.to(device)

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
        
        # Concatenate the history with the new input
        if st.session_state.history:
            bot_input_ids = torch.cat([torch.tensor(st.session_state.history), new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate a response
        chat_history_ids = model.generate(bot_input_ids.to(device), max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode the response
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Display bot response and update conversation history
        st.write(f"AI: {bot_response}")
        st.session_state.history.append(chat_history_ids.tolist())  # Save the new history
