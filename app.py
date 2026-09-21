
import streamlit as st
import pandas as pd

from chatbot_engine import MedicalChatbotEngineV3


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedQuad Medical Chatbot",
    page_icon="🩺",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div.stButton > button {
        width: 100%;
        min-height: 45px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 MedQuad Medical Chatbot")

st.write(
    "Ask questions about medical conditions, symptoms, "
    "treatments, and general health."
)

st.info(
    "This chatbot provides educational information from "
    "its dataset. It is not a substitute for professional "
    "medical advice."
)


# ============================================================
# 2. LOAD CHATBOT
# ============================================================


# ============================================================
# 2. LOAD CHATBOT
# ============================================================

@st.cache_resource(show_spinner="Loading MedQuad chatbot...")
def load_chatbot():
    df = pd.read_csv("medquad.csv")

    chatbot = MedicalChatbotEngineV3(
        dataframe=df,
        threshold=0.30,
        top_k=3,
    )

    return chatbot


try:
    chatbot = load_chatbot()

except Exception as e:
    st.error(f"Failed to load chatbot: {e}")
    st.stop()

# ============================================================
# 3. INITIALIZE SESSION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am the MedQuad Medical Chatbot. "
                "You can ask me medical questions in English."
            ),
        }
    ]


# ============================================================
# 4. QUICK QUESTIONS
# ============================================================

st.subheader("Quick Questions")

quick_questions = [
    "What causes high blood pressure?",
    "How can I prevent heart disease?",
    "What is Q fever?",
    "How is Q fever treated?",
]

cols = st.columns(2)

for i, question in enumerate(quick_questions):
    if cols[i % 2].button(
        question,
        key=f"quick_{i}",
    ):
        st.session_state.pending_question = question
        st.rerun()


# ============================================================
# 5. PROCESS NEW QUESTION BEFORE DISPLAYING CHAT
# ============================================================

if "pending_question" in st.session_state:
    user_text = st.session_state.pop("pending_question")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    try:
        with st.spinner("Generating response..."):
            response = chatbot.get_response(user_text)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

    except Exception as e:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error generating response: {e}",
            }
        )


# ============================================================
# 6. DISPLAY CHAT HISTORY
# ============================================================

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ============================================================
# 7. INPUT AT THE BOTTOM
# ============================================================

st.divider()

with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input(
        "Ask another question",
        placeholder="Type your medical question in English...",
    )

    submitted = st.form_submit_button("Send")

    if submitted and prompt.strip():
        st.session_state.pending_question = prompt.strip()
        st.rerun()


# ============================================================
# 8. CLEAR CHAT
# ============================================================

if st.button("Clear Chat", key="clear_chat"):
    st.session_state.messages = []
    chatbot.reset_conversation()
    st.rerun()
