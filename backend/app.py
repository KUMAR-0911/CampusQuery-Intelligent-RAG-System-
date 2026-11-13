"""
app.py — Streamlit web interface for the College Assistant chatbot.

HOW TO RUN
----------
    cd backend
    streamlit run app.py

WHAT IT DOES
------------
  • MAIN AREA     — Ask questions about the college's preloaded documents.
                    The bot retrieves relevant chunks from Qdrant and
                    generates an answer using Groq.
  • HEADER        — Clear the current conversation.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from RETRIVAL.pipeline import answer_question, get_vector_store


def answer_user_question(question: str) -> dict[str, Any]:
    """Retrieve relevant chunks and generate an answer for the chat.

    Passes the last few messages as conversation context so the LLM
    can follow multi-turn conversations.
    """
    print(f"[streamlit] Processing chat question ({len(question)} chars) …")

    # Send the chat history (minus the current question) as context.
    conversation_history = st.session_state.messages[:-1]

    result = answer_question(
        question,
        get_vector_store(),
        conversation_context={"recent_messages": conversation_history},
    )

    print("[streamlit] Chat response ready.")
    return result


# ═══════════════════════════════════════════════════════════════
#  Streamlit page layout
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="College Assistant", page_icon="🎓")

# Initialise chat history in session state.
st.session_state.setdefault("messages", [])

# ── Header: assistant name and chat controls ───────────────────

title_column, clear_column = st.columns([8, 1])
with title_column:
    st.title("🎓 College Assistant")
    st.caption("Ask questions and get answers from your college's documents.")
with clear_column:
    if st.button("🗑️", help="Clear chat", type="tertiary"):
        st.session_state.messages = []


# ── Main area: chat interface ───────────────────────────────────

# Render all previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input.
question = st.chat_input("Ask a question about your documents …")

if question:
    print("[streamlit] User submitted a question.")

    # Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display the assistant's response.
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer …"):
                result = answer_user_question(question)

            # Display the answer.
            st.markdown(result["answer"])

        # Save the assistant's reply to chat history.
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )

    except Exception as error:
        print(f"[streamlit] Chat failed: {type(error).__name__}")
        st.error(f"❌ Could not answer the question: {error}")
