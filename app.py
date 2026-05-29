# Import semuanya dulu
import streamlit as st
# Asumsikan file 'bot' ada
from bot import build_agent

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="PromptCraft Bot | Buat Prompt AI yang Sempurna",
    page_icon="✨",
    layout="wide"
)

st.title("✨ PromptCraft Bot: Buat Prompt AI yang Lengkap & Efektif!")

st.markdown("---")

st.markdown(
    """
    **Asisten Prompt Engineering** yang siap membantumu. Tanyakan apa pun tentang:
    * **Struktur Prompt** (Role, Context, Task, Format, Constraints)
    * **Teknik Prompting** (Zero-shot, Few-shot, Chain-of-Thought, Tree-of-Thought)
    * **Optimasi Prompt** untuk berbagai model AI (GPT, Claude, Gemini, dll.)
    * **Contoh Prompt** siap pakai untuk berbagai kebutuhan (coding, menulis, analisis, dll.)
    """
)
st.markdown("---")


# Session state
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent

# --- UI dan Tombol Reset ---

col1, col2 = st.columns([1, 5])
with col1:
    reset_chat_button = st.button("🔄 Mulai Sesi Baru", help="Klik untuk menghapus riwayat obrolan dan memulai pertanyaan baru.")
    if reset_chat_button:
        st.session_state.messages = []
        st.rerun()


# Tampilkan riwayat pesan
for m in st.session_state.messages:
    role_icon = "user" if m["role"] == "user" else "assistant" if m["role"] == "assistant" else "⚙️"

    with st.chat_message(role_icon):
        st.markdown(m["content"], unsafe_allow_html=True)


# Input pengguna
user_input = st.chat_input("Contoh: 'Buatkan prompt untuk menulis artikel SEO' atau 'Jelaskan teknik Chain-of-Thought'")


if user_input is not None:
    # 1. Simpan dan tampilkan input pengguna
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Proses respons agent
    with st.spinner("Merancang prompt terbaik untuk kamu... ✨ Menyusun struktur & teknik yang tepat..."):
        ai_output = ""

        # Streaming langkah-langkah agent
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            for step in agent.stream({"input": user_input}):
                if "actions" in step.keys():
                    for action in step["actions"]:
                        tool_name = action.tool
                        tool_input = action.tool_input

                        # Styling Tool
                        tool_message = f"""
                        <div style="border-left: 4px solid #7C3AED; padding: 6px 10px; background-color: #F5F3FF; border-radius: 4px; font-size: 13px; color: #555555; margin: 4px 0;">
                            ⚙️ **Aksi Tool:** <code>{tool_name}</code> (Input: {tool_input})
                        </div>
                        """
                        st.session_state.messages.append({
                            "role": "⚙️",
                            "content": tool_message,
                        })
                        st.markdown(tool_message, unsafe_allow_html=True)
                        message_placeholder = st.empty()

                if "output" in step.keys():
                    ai_output = step["output"]

            message_placeholder.markdown(ai_output)

            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_output,
            })
