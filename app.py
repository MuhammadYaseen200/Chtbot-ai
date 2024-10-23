import streamlit as st
from transformers import pipeline

# Load the model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define the context for the chatbot
context = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines that are designed to think and act like humans. 
AI can be classified into narrow AI, which is designed to perform a narrow task (like facial recognition or internet searches), 
and general AI, which has the ability to understand, learn, and apply intelligence broadly, similar to a human.
Applications of AI include natural language processing, robotics, computer vision, and more.
"""

# Streamlit app
st.title("AI Chatbot")

# Get user input
user_input = st.text_input("Ask me anything about Artificial Intelligence:")

if user_input:
    answer = qa_pipeline(question=user_input, context=context)
    st.write(f"**Answer:** {answer['answer']}")
