import streamlit as st

from groq import Groq

import os

from dotenv import load_dotenv



# Load environment variables

load_dotenv(dotenv_path=".env")



# Set up the page

st.set_page_config(page_title="My AI Assistant", page_icon="🤖")

st.title("🤖 My AI Assistant")

st.caption("Powered by Llama 3.3 via Groq")



# Initialize the Groq client

api_key = os.getenv("GROQ_API_KEY")

# Debug: Check if key is loaded (remove this after testing)

if not api_key or api_key == "your_groq_api_key_here":

    client = None

    st.warning("⚠️ Please set your GROQ_API_KEY in the .env file")

else:

    try:

        client = Groq(api_key=api_key)

    except Exception as e:

        client = None

        st.error(f"❌ Error initializing Groq client: {str(e)}")



# Create a place to store chat history

if "messages" not in st.session_state:

    st.session_state.messages = []



# Display all previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])



# Get user input

user_input = st.chat_input("Ask me anything...")



if user_input:

    # Add user message to history and display it

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):

        st.write(user_input)

    

    # Get AI response

    with st.chat_message("assistant"):

        if client is None:

            st.error("❌ API key not configured. Please set GROQ_API_KEY in your .env file.")

        else:

            with st.spinner("Thinking..."):

                try:

                    # Call Groq API with Llama model

                    response = client.chat.completions.create(

                        model="llama-3.3-70b-versatile",

                        messages=st.session_state.messages,

                        max_tokens=1000,

                        temperature=0.7

                    )

                    

                    # Extract the response text

                    assistant_response = response.choices[0].message.content

                    

                    # Display and save the response

                    st.write(assistant_response)

                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                except Exception as e:

                    st.error(f"❌ Error: {str(e)}")

