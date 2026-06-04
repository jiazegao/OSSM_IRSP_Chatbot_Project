import streamlit as st
import json
import time
from openai import OpenAI

# API Settings --------------------------------------------------
client = OpenAI(base_url = "https://api.deepseek.com",
                api_key = "sk-5a2343a17c5345669c3969a781d922b0")

# Multi-Chat Management ------------------------------------------------

# Initialize chats -----------
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = -1

# Create a new chat -------
def new_chat(switch = True):
    new_id = len(st.session_state.chats)
    st.session_state.chats.append({
        "id": new_id,
        "title": "New conversation",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
        "stream": None,
        "streaming": False,
        "temp_bot_ui": None
    })
    if switch:
        st.session_state.current_chat_id = new_id

# Retrieve chat history
if len(st.session_state.chats) == 0:
    with open("chat_history.json", "r", encoding="utf-8") as file:
        try:
            st.session_state.chats = json.load(file)
            st.session_state.current_chat_id = 0
        except json.decoder.JSONDecodeError:
            st.session_state.chats = []
            st.session_state.current_chat_id = -1
            new_chat(True)

# Sidebar content ---------
with st.sidebar:
    st.button("➕ New chat", use_container_width=True, on_click=new_chat)

    st.markdown("---")
    st.caption("Your conversations")

    # Show existing chats, highlight current
    for chat in st.session_state.chats:
        btn_label = chat["title"]
        if chat["id"] == st.session_state.current_chat_id:
            btn_label = f"**{btn_label}**"
        if st.button(btn_label, key=chat["id"], use_container_width=True):
            st.session_state.current_chat_id = chat["id"]
            st.rerun()

# RAG Settings -------------------------------------------------
textbooks = []
textbook_paths = ["Textbooks/CompArch.txt",
                  "Textbooks/USHist.txt"]
subject_titles = ["Computer Architecture",
                  "US History"]
RAG_IDs = {}
RAG_Settings = {"Computer Architecture": {"temp": 0.0, "top_p": 1.0, "RE": "high", "TM": "enabled"},
                "US History": {"temp": 0.0, "top_p": 1.0}}

def getRAGPreset(item, id):
    global RAG_IDs, RAG_Settings
    if id in RAG_IDs.keys() and item in RAG_IDs[id].keys():
        return RAG_Settings[RAG_IDs[id]][item]
    if item == "temp":
        return 0.5
    if item == "top_p":
        return 1.0
    if item == "TM":
        return "disabled"
    return None

if "RAGInit" not in st.session_state:
    st.session_state.RAGInit = True

    initialized = False
    for c in st.session_state.chats:
        if c["title"] == "Computer Architecture" or c["title"] == "US History":
            initialized = True
            break
    if not initialized:
        for path in textbook_paths:
            with open(path, "r", encoding="utf-8") as file:
                textbooks.append(file.read())

        for i in range(len(textbooks)):
            new_chat(switch = False)
            st.session_state.chats[-1]["messages"].append({"role": "system", "content": textbooks[i]})
            st.session_state.chats[-1]["title"] = subject_titles[i]
            RAG_IDs[st.session_state.chats[-1]["id"]] = st.session_state.chats[-1]["title"]
        st.rerun()


# Current Chat Management ---------------------------------------------
curr_id = st.session_state.current_chat_id
curr_chat = st.session_state.chats[curr_id]

# Title ----------------
st.title(curr_chat["title"])

# Display chat messages from history -------
count = 0
mes = None
for message in curr_chat["messages"]:
    count += 1
    if count < len(curr_chat["messages"]) or message["role"] != "assistant" or not curr_chat["streaming"]:
        if message["role"] != "system":
            mes = st.chat_message(message["role"])
            mes.markdown(message["content"])
    else:
        curr_chat["temp_bot_ui"] = st.empty()
        if (curr_chat["messages"][-1]["content"] == ""):
            curr_chat["temp_bot_ui"].chat_message("assistant").markdown("...")
        else:
            curr_chat["temp_bot_ui"].chat_message("assistant").markdown(curr_chat["messages"][-1]["content"])

# Accept user input -------
if prompt := st.chat_input("What is up?"):
    # Display user message
    user_input = st.chat_message("user")
    user_input.markdown(prompt)
    # Record user message
    curr_chat["messages"].append({"role": "user", "content": prompt})

    curr_chat["stream"] = client.chat.completions.create(
        model = "deepseek-v4-flash",
        messages = curr_chat["messages"],
        stream = True,
        reasoning_effort=getRAGPreset("RE", curr_id),
        temperature=getRAGPreset("temp", curr_id),
        top_p=getRAGPreset("top_p", curr_id),
        extra_body = {
            "thinking": {
                "type": getRAGPreset("TM", curr_id)
            }
        }
    )
    curr_chat["streaming"] = True
    curr_chat["messages"].append({"role": "assistant", "content": ""})

    # Animation for waiting
    temp_ui = st.empty()
    curr_chat["temp_bot_ui"] = temp_ui
    temp_ui.chat_message("assistant").markdown(".")
    time.sleep(0.8)
    temp_ui.empty()
    temp_ui.chat_message("assistant").markdown("..")
    time.sleep(0.8)
    temp_ui.empty()
    temp_ui.chat_message("assistant").markdown("...")

# Update output ---------
just_finished = False
while curr_chat["streaming"] and curr_chat["stream"] is not None:
    try:
        chunk = next(curr_chat["stream"])
        delta = chunk.choices[0].delta.content
        if delta:
            curr_chat["messages"][-1]["content"] += delta
            curr_chat["temp_bot_ui"].empty()
            curr_chat["temp_bot_ui"].chat_message("assistant").markdown(curr_chat["messages"][-1]["content"] + "█")
            if curr_chat["title"] == "New conversation":
                user_msgs = [m for m in curr_chat["messages"] if m["role"] == "user"]
                if user_msgs:
                    curr_chat["title"] = user_msgs[-1]["content"][:28]
    except StopIteration:
        curr_chat["stream"] = None
        curr_chat["streaming"] = False
        just_finished = True
    time.sleep(0.02)

# Update chat history
if just_finished:
    with open("chat_history.json", "w", encoding="utf-8") as file:
        print (st.session_state.chats)
        file.write(json.dumps(st.session_state.chats, ensure_ascii=False, default=lambda o: None))
    st.rerun()