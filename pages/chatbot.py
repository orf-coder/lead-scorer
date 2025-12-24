import streamlit as st
from Chatbot.bot1 import LeadScorerBot

# Page configuration
st.set_page_config(
    page_title="Lead Scorer Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Lead Scorer Chatbot")
st.markdown("Chat with the AI assistant for lead scoring help.")

# Initialize bot in session state
if 'bot' not in st.session_state:
    st.session_state.bot = LeadScorerBot()

# Initialize messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = "Hello! I'm your AI lead scoring assistant. To get started, please enter the lead message you'd like to score. I'll then guide you through collecting the additional details (name, company, job title, source) step by step."
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        response = st.session_state.bot.process_message(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})