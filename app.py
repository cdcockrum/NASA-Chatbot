import os
import re
import requests
import torch
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from langdetect import detect  # Ensure this package is installed

# ✅ Check for GPU or Default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")  # Debugging info

# ✅ Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is not set. Please add it to your environment variables.")

NASA_API_KEY = os.getenv("NASA_API_KEY")
if NASA_API_KEY is None:
    raise ValueError("NASA_API_KEY is not set. Please add it to your environment variables.")

# ✅ Set Up Streamlit
st.set_page_config(page_title="HAL - NASA ChatBot", page_icon="🚀")

# ✅ Initialize Session State Variables (Ensuring Chat History Persists)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# ✅ Initialize Hugging Face Model (Explicitly Set to CPU/GPU)
def get_llm_hf_inference(model_id="meta-llama/Llama-2-7b-chat-hf", max_new_tokens=800, temperature=0.3):
    return HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,  # 🔥 Lowered temperature for more factual and structured responses
        token=HF_TOKEN,
        task="text-generation",
        device=-1 if device == "cpu" else 0  # ✅ Force CPU (-1) or GPU (0)
    )

# ✅ Ensure English Responses
def ensure_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang != "en":
            return "⚠️ Sorry, I only respond in English. Can you rephrase your question?"
    except:
        return "⚠️ Language detection failed. Please ask your question again."
    return text

# ✅ Main Response Function (Fixing Repetition & Context)
def get_response(system_message, chat_history, user_text, max_new_tokens=800):
    # ✅ Ensure conversation history is included correctly
    filtered_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in chat_history[-5:]  # ✅ Only keep the last 5 exchanges to prevent overflow
    )
 
    prompt = PromptTemplate.from_template(
        "[INST] You are a highly knowledgeable AI assistant. Answer concisely, avoid repetition, and structure responses well."
        "\n\nCONTEXT:\n{chat_history}\n"
        "\nLATEST USER INPUT:\nUser: {user_text}\n"
        "\n[END CONTEXT]\n"
        "Assistant:"
    )

    # ✅ Invoke Hugging Face Model
    hf = get_llm_hf_inference(max_new_tokens=max_new_tokens, temperature=0.3)  # 🔥 Lowered temperature
    chat = prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key='content')

    response = chat.invoke(input=dict(system_message=system_message, user_text=user_text, chat_history=filtered_history))
    
    # Clean up the response - remove any "HAL:" prefix if present
    response = response.split("HAL:")[-1].strip() if "HAL:" in response else response.strip()
    response = ensure_english(response)

    if not response:
        response = "I'm sorry, but I couldn't generate a response. Can you rephrase your question?"

    # ✅ Update conversation history
    chat_history.append({'role': 'user', 'content': user_text})
    chat_history.append({'role': 'assistant', 'content': response})

    # ✅ Keep only last 10 exchanges to prevent unnecessary repetition
    return response, chat_history[-10:]

# ✅ Streamlit UI
st.title("🚀 HAL - NASA AI Assistant")

# ✅ Justify all chatbot responses
st.markdown("""
    <style>
    .user-msg, .assistant-msg {
        padding: 11px;
        border-radius: 10px;
        margin-bottom: 5px;
        width: fit-content;
        max-width: 80%;
        text-align: justify;
    }
    .user-msg { background-color: #696969; color: white; }
    .assistant-msg { background-color: #333333; color: white; }
    .container { display: flex; flex-direction: column; align-items: flex-start; }
    @media (max-width: 600px) { .user-msg, .assistant-msg { font-size: 16px; max-width: 100%; } }
    </style>
""", unsafe_allow_html=True)

# ✅ Chat UI
user_input = st.chat_input("Type your message here...")

if user_input:
    # Get response and update chat history
    response, st.session_state.chat_history = get_response(
        system_message="You are a helpful AI assistant.",
        user_text=user_input,
        chat_history=st.session_state.chat_history
    )

# ✅ Display chat history (ONLY display from history, not separately)
st.markdown("<div class='container'>", unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='user-msg'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-msg'><strong>HAL:</strong> {message['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
