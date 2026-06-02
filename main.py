import streamlit as st
import numpy as np
import random
import time
from openai import OpenAI

client = OpenAI(base_url = "https://api.deepseek.com",
                api_key = st.secrets["OPENAI_API_KEY"])

st.title("ChatGPT(???)")

# Initialize chats
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = -1
if "streaming" not in st.session_state:
    st.session_state.streaming = False

def new_chat():
    new_id = len(st.session_state.chats)
    st.session_state.chats.append({
        "id": new_id,
        "title": "New conversation",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
        "stream": None,
        "response": "",
    })
    if not st.session_state.streaming:
        st.session_state.current_chat_id = new_id

# Sidebar content
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        new_chat()

    st.markdown("---")
    st.caption("Your conversations")

    # Show existing chats, highlight current
    for chat in st.session_state.chats:
        btn_label = chat["title"]
        if chat["id"] == st.session_state.current_chat_id:
            btn_label = f"**{btn_label}**"
            curr_chat = chat
        if st.button(btn_label, key=chat["id"], use_container_width=True):
            if not st.session_state.streaming:
                st.session_state.current_chat_id = chat["id"]
                st.rerun()

curr_id = st.session_state.current_chat_id
if (len(st.session_state.chats) == 0):
    new_chat()
    st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.chats[curr_id]["messages"]:
    if (message["role"] == "user"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Record user message
    st.session_state.chats[curr_id]["messages"].append({"role": "user", "content": prompt})

# Display assistant response
if prompt:
    st.session_state.chats[curr_id]["stream"] = client.chat.completions.create(
        model = "deepseek-v4-flash",
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chats[curr_id]["messages"]
        ],
        stream = True,
        reasoning_effort="low",
        temperature=0.5
    )

# Detect if still transmitting
if not st.session_state.streaming and st.session_state.chats[curr_id]["stream"]:
    st.session_state.streaming = True
    st.session_state.chats[curr_id]["response"] = st.write_stream(st.session_state.chats[curr_id]["stream"])
    st.session_state.streaming = False
    if st.session_state.chats[curr_id]["response"]:
        st.session_state.chats[curr_id]["messages"].append({"role": "assistant", "content": st.session_state.chats[curr_id]["response"]})
    print(st.session_state.chats[curr_id]["response"])