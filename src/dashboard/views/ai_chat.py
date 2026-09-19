"""
AI Crime Analytics Assistant Chat View
Provides a glassmorphic chat interface powered by OpenAI.
"""

import pandas as pd
import streamlit as st

from src.dashboard.ai_assistant import generate_chat_response, get_openai_client
from src.dashboard.theme import (
    BG_CARD,
    BG_DARK,
    BG_GLASS,
    BORDER_GLASS,
    PRIMARY,
    PRIMARY_MUTED,
    SECONDARY,
    SHADOW_GLASS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def render(
    df_clusters: pd.DataFrame, 
    df_metro: pd.DataFrame,
    df_profiles: pd.DataFrame,
    df_murder: pd.DataFrame,
    df_cyber: pd.DataFrame
):
    """Render the AI Chatbot interface."""

    # ── Page Header ──
    st.markdown(
        f"""
        <div style="padding: 10px 0 24px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:24px;">🤖</span>
                <h1 style="font-size:30px; font-weight:800; color:{TEXT_PRIMARY}; margin:0; letter-spacing:-0.5px;">
                AI Intelligence Assistant</h1>
            </div>
            <p style="color:{TEXT_SECONDARY}; font-size:14px; margin:0; line-height:1.5;">
            Ask questions about spatial typologies, district profiles, metropolitan policing, and anomalies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the NCRB 2024 Crime Analytics AI. Ask me about district crime patterns, typologies, policing quadrants, or anomaly detection."}
        ]

    client = get_openai_client()

    # ── Configuration Panel (API Key) ──
    with st.expander("⚙️ Connection Settings (OpenAI API Key)", expanded=(client is None)):
        st.markdown(
            f"""
            <div style="color:{TEXT_SECONDARY}; font-size:13px; margin-bottom:12px;">
            Enter your OpenAI API key to activate the assistant. The key is only stored in your current session memory.
            </div>
            """,
            unsafe_allow_html=True,
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=st.session_state.get("openai_api_key", ""),
            help="Your API key from platform.openai.com",
            key="openai_api_key"
        )
        if api_key_input:
            client = get_openai_client()
            if client:
                st.success("API Key activated. The assistant is ready.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── Suggestion Chips ──
    st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:12px; margin-bottom:8px; font-weight:600;'>SUGGESTED PROMPTS</div>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    suggestions = [
        "What are the 5 crime typologies?",
        "Compare Q1 and Q3 policing quadrants.",
        "Which districts are DBSCAN anomalies?",
        "Top motives for murder in metros?"
    ]
    
    # Store the selected suggestion in session state to feed into chat input
    for i, (col, sug) in enumerate(zip(chip_cols, suggestions)):
        if col.button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["prompt_input"] = sug

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # ── Chat Interface ──
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    # If a suggestion was clicked, we use it. Otherwise, standard chat input.
    prompt = st.chat_input("Ask about NCRB 2024 Crime Analytics...")
    
    # If a suggestion was clicked, process it as a prompt, then clear it so it doesn't loop
    if "prompt_input" in st.session_state and st.session_state["prompt_input"]:
        prompt = st.session_state["prompt_input"]
        st.session_state["prompt_input"] = None

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            if client is None:
                st.error("Please provide a valid OpenAI API Key in the settings above.")
                return
            
            message_placeholder = st.empty()
            # Stream response
            response_stream = generate_chat_response(
                client=client, 
                messages=st.session_state.messages,
                df_clusters=df_clusters,
                df_metro=df_metro
            )
            
            full_response = ""
            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
