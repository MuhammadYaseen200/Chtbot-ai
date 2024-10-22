import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pretrained model and tokenizer from Hugging Face (using distilGPT-2 as an example)
model_name = "distilgpt2"  # A smaller and free version of GPT-2
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
        num_return_sequences=1,  # Generate 1 response
        no_repeat_ngram_size=2,   # Prevent repetition
        do_sample=True,           # Sampling to create varied responses
        temperature=0.7,          # Control creativity (lower is more deterministic)
        top_p=0.9                 # Control diversity (higher values create more diverse answers)
    )
    
    # Decode the generated tokens and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app layout
def main():
    st.title("AI Chatbot")
    st.write("Ask anything about Artificial Intelligence!")
    
    # Input box for user queries
    user_input = st.text_input("You: ", "")
    
    if user_input:
        # Generate and display the response
        response = generate_response(user_input)
        st.write(f"AI Chatbot: {response}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
  
