import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pretrained model and tokenizer from Hugging Face
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response from the model
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response from the model
    outputs = model.generate(
        inputs, 
        max_length=150, 
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app layout
def main():
    st.title("AI Chatbot")
    st.write("Ask anything about Artificial Intelligence!")

    # Input field managed by session_state
    if 'input' not in st.session_state:
        st.session_state.input = ""

    user_input = st.text_input("You: ", st.session_state.input)

    # Generate response and display
    if user_input:
        response = generate_response(user_input)
        st.write(f"AI Chatbot: {response}")
    
    # Button to clear chat
    if st.button("Clear Chat"):
        st.session_state.input = ""  # Clear input

if __name__ == "__main__":
    main()
    
