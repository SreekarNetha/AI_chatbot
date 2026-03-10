import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page config
st.set_page_config(page_title="OpenChat Assistant", page_icon="🤖", layout="wide")

st.title("🤖 OpenChat Local Assistant")

# Sidebar
st.sidebar.header("Settings")
max_tokens = st.sidebar.slider("Max Tokens", 100, 5000, 1000)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            offload_folder="offload",
            trust_remote_code=True,
            revision="main"
        )
    except Exception as e:
        st.warning(f"GPU loading failed: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            trust_remote_code=True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


tokenizer, model, device = load_model()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something...")

if user_input:

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # Encode
        input_ids = tokenizer.encode(
            user_input,
            return_tensors="pt"
        ).to(device)

        # Generate
        output = model.generate(
            input_ids,
            max_length=max_tokens,
            do_sample=True,
            temperature=temperature
        )

        decoded_output = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        # Remove prompt from response
        response = decoded_output.replace(user_input, "").strip()

        placeholder.markdown(response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )