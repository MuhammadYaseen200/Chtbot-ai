# Streamlit app layout
def main():
    st.title("AI Chatbot")
    st.write("Ask anything about Artificial Intelligence!")

    # Input field managed by session_state
    if 'input' not in st.session_state:
        st.session_state.input = ""

    user_input = st.text_input("You: ", st.session_state.input)

    # Generate and display the response if input is provided
    if user_input:
        response = generate_response(user_input)
        if response is not None:  # Check for a valid response
            st.write(f"AI Chatbot: {response}")
        else:
            st.write("AI Chatbot: Sorry, I couldn't generate a response.")
    
    # Button to clear chat
    if st.button("Clear Chat"):
        st.session_state.input = ""  # Clear input

if __name__ == "__main__":
    main()
