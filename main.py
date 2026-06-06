import re
import streamlit as st
import json
import time
from openai import OpenAI

# API Settings ----------------------------------------------------
client = OpenAI(base_url = "https://api.deepseek.com",
                api_key = st.secrets["OPENAI_API_KEY"])

# Functions and Constants -------------------------------------------
textbooks = []
textbook_paths = ["Textbooks/CompArch.txt",
                  "Textbooks/USHist.txt",
                  "Textbooks/DS.txt"]
subject_titles = ["Computer Architecture",
                  "US History",
                  "Data Structures"]
RAG_IDs = {}
RAG_Settings = {"Computer Architecture": {"temp": 0.0, "top_p": 1.0, "RE": "high", "TM": "enabled"},
                "US History": {"temp": 0.0, "top_p": 1.0, "RE": "high", "TM": "enabled"},
                "Data Structures": {"temp": 0.0, "top_p": 1.0, "RE": "high", "TM": "enabled"}}

# Create a new chat
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

def extract_text_from_file(file) -> str:
    from pypdf import PdfReader
    import io
    if file.name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return file.read().decode("utf-8")

# Retrieve RAG properties
def get_rag_preset(item, id):
    global RAG_IDs, RAG_Settings
    if id in RAG_IDs.keys() and item in RAG_IDs[id].keys():
        return RAG_Settings[RAG_IDs[id]][item]
    if item == "temp":
        return 0.3
    if item == "top_p":
        return 1.0
    if item == "RE":
        return "high"
    if item == "TM":
        return "enabled"
    return None

# Cleanup latex notation
def get_cleaned_latex(string):
    if not isinstance(string, str):
        return string
    string = re.sub(r'\\{1,2}\[(.*?)\\{1,2}\]', r'$$\1$$', string, flags=re.DOTALL)
    string = re.sub(r'\(\((.*?)\)\)', r'$\1$', string, flags=re.DOTALL)
    string = re.sub(r'\\{1,2}\((.*?)\\{1,2}\)', r'$\1$', string, flags=re.DOTALL)
    return string

@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )

def build_vectorstore_from_text(text: str):
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    return FAISS.from_documents(chunks, get_embeddings())

def get_rag_context(query: str, vectorstores, k: int = 3) -> str:
    all_docs = []
    for fname, store in vectorstores.items():
        docs = store.similarity_search(query, k=k)
        for doc in docs:
            # Tag each chunk with its source file
            all_docs.append(f"[{fname}]\n{doc.page_content}")
    return "\n\n---\n\n".join(all_docs)

def get_chat_title(text: str):
    return  client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "system", "content": "Generate a title for this conversation in 3 words or less. " +
                                                "Return only the title, no quotes, no punctuation, no explanation. Prompt: " + text}],
        stream=False,
        temperature=0.5,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    ).choices[0].message.content

# Multi-Chat Management ------------------------------------------------

# Initialize chats
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = -1
if "last_switch_time" not in st.session_state:
    st.session_state.last_switch_time = time.time()

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

# Sidebar content
with st.sidebar:
    st.title("Conversations")
    st.button("➕ New chat", use_container_width=True, on_click=new_chat)

    # Subject-specific chats
    st.markdown("---")
    st.caption("RAG-Assisted Chats")

    # Show existing chats, highlight current
    for chat in st.session_state.chats:
        if chat["title"] in subject_titles:
            btn_label = chat["title"]
            if chat["id"] == st.session_state.current_chat_id:
                btn_label = f"**{btn_label}**"
            if st.button(btn_label, key=chat["id"], use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.last_switch_time = time.time()
                st.rerun()

    # Personal chats
    st.markdown("---")
    st.caption("Your Conversations")

    # Show existing chats, highlight current
    for chat in st.session_state.chats:
        if chat["title"] not in subject_titles:
            btn_label = chat["title"]
            if chat["id"] == st.session_state.current_chat_id:
                btn_label = f"**{btn_label}**"
            if st.button(btn_label, key=chat["id"], use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.last_switch_time = time.time()
                st.rerun()

    curr_id = st.session_state.current_chat_id
    curr_chat = st.session_state.chats[curr_id]

    # Build and cache vectorstore in session state per chat
    vs_key = f"vectorstore_{curr_id}"
    if vs_key not in st.session_state:
        st.session_state[vs_key] = {}

    # File management
    if vs_key in st.session_state and st.session_state[vs_key]:
        st.markdown("---")
        st.caption(f"📎 Files for \"{curr_chat['title']}\"")
        for fname in list(st.session_state[vs_key].keys()):
            col1, col2 = st.columns([3, 1])
            col1.caption(f"📄 {fname[:20]}...")
            if col2.button("❌", key=f"remove_{curr_id}_{fname}"):
                del st.session_state[vs_key][fname]
                st.rerun()

# RAG Initialization  -------------------------------------------------
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

# Title
st.title(curr_chat["title"])

# Display chat messages from history
count = 0
mes = None
for message in curr_chat["messages"]:
    count += 1
    if count < len(curr_chat["messages"]) or message["role"] != "assistant" or not curr_chat["streaming"]:
        if message["role"] != "system":
            mes = st.chat_message(message["role"])
            mes.markdown(get_cleaned_latex(message["content"]))
    else:
        try:
            curr_chat["temp_bot_ui"].empty()
        except:
            curr_chat["temp_bot_ui"] = st.empty()
        if (curr_chat["messages"][-1]["content"] == ""):
            curr_chat["temp_bot_ui"].chat_message("assistant").markdown("...")
        else:
            curr_chat["temp_bot_ui"].chat_message("assistant").markdown(get_cleaned_latex(curr_chat["messages"][-1]["content"]))

# Accept user input
if chat := st.chat_input("What is up?", accept_file="multiple", file_type=["txt", "pdf"]):
    prompt = chat.text
    uploaded_files = chat.files  # list of uploaded files

    if not prompt:
        st.error("Please enter a message")
    else:
        user_input = st.chat_message("user")
        user_input.markdown(prompt)
        curr_chat["messages"].append({"role": "user", "content": prompt})

        # Index any newly added files
        if uploaded_files:
            new_files = False
            for file in uploaded_files:
                if file.name not in st.session_state[vs_key]:
                    with st.spinner(f"Indexing {file.name}..."):
                        text = extract_text_from_file(file)  # ← replaces file.read().decode()
                        st.session_state[vs_key][file.name] = build_vectorstore_from_text(text)
                    new_files = True
                else:
                    st.error("Document exists! Proceeding...")
            if new_files:
                st.success("Documents indexed!")

        # ── LangChain injection (personal chats only) ──────────────
        vs_key = f"vectorstore_{curr_id}"
        messages_to_send = curr_chat["messages"]  # default: send as-is

        if vs_key in st.session_state and st.session_state[vs_key]:
            rag_context = get_rag_context(prompt, st.session_state[vs_key])
            messages_to_send = (
                    curr_chat["messages"][:-1]
                    + [{"role": "system", "content": f"Relevant document context:\n{rag_context}"}]
                    + [curr_chat["messages"][-1]]
            )

        # Animation for waiting
        temp_ui = st.empty()
        curr_chat["temp_bot_ui"] = temp_ui
        temp_ui.chat_message("assistant").markdown(".")
        time.sleep(0.8)
        temp_ui.empty()
        temp_ui.chat_message("assistant").markdown("..")

        curr_chat["stream"] = client.chat.completions.create(
            model = "deepseek-v4-flash",
            messages = messages_to_send,
            stream = True,
            reasoning_effort=get_rag_preset("RE", curr_id),
            temperature=get_rag_preset("temp", curr_id),
            top_p=get_rag_preset("top_p", curr_id),
            extra_body = {
                "thinking": {
                    "type": get_rag_preset("TM", curr_id)
                }
            }
        )

        curr_chat["streaming"] = True
        curr_chat["messages"].append({"role": "assistant", "content": ""})
        temp_ui.empty()
        temp_ui.chat_message("assistant").markdown("...")

        if curr_chat["title"] == "New conversation":
            curr_chat["title"] = get_chat_title(curr_chat["messages"][-2]["content"])
        st.rerun()

# Retrieve from stream
just_finished = False
while curr_chat["streaming"] and curr_chat["stream"] is not None:
    try:
        chunk = next(curr_chat["stream"])
        delta = chunk.choices[0].delta.content
        if delta:
            curr_chat["messages"][-1]["content"] += delta
            curr_chat["temp_bot_ui"].empty()
            curr_chat["temp_bot_ui"].chat_message("assistant").markdown(get_cleaned_latex(curr_chat["messages"][-1]["content"]) + "█")
    except StopIteration:
        curr_chat["stream"] = None
        curr_chat["streaming"] = False
        just_finished = True
    # Fastforward if just switched chat
    if time.time() - st.session_state.last_switch_time > 3.0:
        time.sleep(0.015)

# Update chat history
if just_finished:
    with open("chat_history.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(st.session_state.chats, ensure_ascii=False, default=lambda o: None))
    st.rerun()