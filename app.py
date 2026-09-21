
import streamlit as st
import pandas as pd

# Ganti import ini sesuai nama file tempat class model kamu berada.
from chatbot_engine import MedicalChatbotEngineV3


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="MedBot v3 - Medical Assistant",
    page_icon="🏥",
    layout="centered"
)


# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f0f4ff;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    .header {
        background: linear-gradient(
            135deg, #1a1f5e, #2d3a8c, #1565c0
        );
        padding: 25px;
        border-radius: 16px;
        color: white;
        margin-bottom: 18px;
    }

    .header h2 {
        color: white;
        margin-bottom: 8px;
    }

    .header p {
        color: #e4edff;
        margin-bottom: 0;
    }

    .disclaimer {
        background: #fff8e1;
        padding: 12px 15px;
        border-left: 4px solid orange;
        border-radius: 8px;
        margin-bottom: 18px;
        color: #594515;
        font-size: 13px;
    }

    div.stButton > button {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET AND MODEL
# =====================================================

@st.cache_resource
def load_chatbot():
    df = pd.read_csv("medquad.csv")

    bot = MedicalChatbotEngineV3(df)

    return bot


try:
    bot = load_chatbot()
except Exception as e:
    st.error(f"Failed to load the chatbot: {e}")
    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="header">
    <h2>🏥 MedBot v3 — Smart Medical Assistant</h2>
    <p>🟢 Context-Aware Semantic Engine</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# DISCLAIMER
# =====================================================

st.markdown("""
<div class="disclaimer">
    ⚠️ <b>Educational use only.</b>
    This chatbot does not replace professional medical advice,
    diagnosis, or treatment. For medical emergencies, contact
    your local emergency services.
</div>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# RESPONSE FUNCTION
# =====================================================

def get_bot_response(question):
    try:
        response = bot.get_response(question)

        if response is None:
            return (
                "Sorry, I could not find a suitable answer "
                "in the medical dataset."
            )

        return str(response)

    except Exception as e:
        return f"An error occurred: {e}"


# =====================================================
# SEND QUESTION
# =====================================================

def send_question(question):
    question = question.strip()

    if not question:
        return

    # Display and save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("MedBot is typing..."):
            response = get_bot_response(question)

        st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


# =====================================================
# QUICK QUESTIONS
# =====================================================

st.markdown("### 💡 Quick Questions")

quick_questions = [
    "What are the symptoms of diabetes?",
    "What causes high blood pressure?",
    "What are the symptoms of asthma?",
    "What are the symptoms of heart disease?",
    "What are common mental health conditions?"
]

cols = st.columns(2)

for i, question in enumerate(quick_questions):
    with cols[i % 2]:
        if st.button(
            question,
            key=f"quick_{i}",
            use_container_width=True
        ):
            send_question(question)
            st.rerun()


# =====================================================
# CHAT INPUT
# =====================================================

user_question = st.chat_input(
    "Type your medical question in English..."
)

if user_question:
    send_question(user_question)
    st.rerun()


# =====================================================
# CLEAR CHAT
# =====================================================

if st.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []

    if hasattr(bot, "conversation_history"):
        bot.conversation_history = []

    st.rerun()


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div style="text-align:center; color:#64748b; font-size:11px;">
    MedBot v3 · Medical information retrieved from the MedQuad dataset
</div>
""", unsafe_allow_html=True)
