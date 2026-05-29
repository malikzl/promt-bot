from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Replicate
from langchain_core.tools import tool

from dotenv import load_dotenv
import requests


def parse_input(input_str):
    parts = input_str.split(";")
    return dict(part.split("=") for part in parts)


@tool
def prompt_technique_tip(input: str) -> str:
    """Get a random tip or technique about AI prompt engineering. The input argument is ignored."""
    try:
        tips = [
            "Zero-shot prompting: Langsung berikan instruksi tanpa contoh. Cocok untuk tugas sederhana.",
            "Few-shot prompting: Berikan 2-3 contoh input/output sebelum pertanyaan utama agar model memahami pola yang diinginkan.",
            "Chain-of-Thought (CoT): Tambahkan 'Jelaskan langkah-langkahmu' atau 'Mari berpikir step by step' untuk meningkatkan akurasi penalaran.",
            "Role prompting: Mulai dengan 'Kamu adalah [peran ahli]...' untuk mendapatkan respons yang lebih kontekstual dan spesifik.",
            "Tree-of-Thought (ToT): Minta model mengeksplorasi beberapa jalur solusi sebelum memilih yang terbaik.",
            "Constraint prompting: Tambahkan batasan eksplisit seperti 'Jawab dalam 3 kalimat' atau 'Gunakan bahasa formal' untuk mengontrol output.",
            "Format prompting: Tentukan format output yang diinginkan, contoh: 'Jawab dalam format JSON', 'Buat dalam bentuk tabel', atau 'Gunakan bullet points'.",
            "Context stuffing: Sertakan konteks latar belakang yang relevan sebelum instruksi utama agar model memiliki pemahaman yang lebih baik.",
            "Iterative refinement: Mulai dengan prompt sederhana, lalu perbaiki secara bertahap berdasarkan output yang didapat.",
            "Negative prompting: Jelaskan apa yang TIDAK ingin kamu dapatkan, contoh: 'Jangan gunakan jargon teknis' atau 'Hindari penjelasan yang terlalu panjang'.",
        ]
        import random
        return f"💡 Tips Prompt Engineering: {random.choice(tips)}"

    except Exception as e:
        return f"Something went wrong with the prompt technique tip tool: {e}"


@tool
def generate_prompt_template(input: str) -> str:
    """
    Generate a complete AI prompt template based on a use case.
    Input should be a short description of the use case, e.g. 'menulis artikel blog SEO'.
    """
    try:
        templates = {
            "default": """
**Prompt Template Umum:**
```
[ROLE] Kamu adalah seorang ahli di bidang {bidang}.

[CONTEXT] {konteks_latar_belakang}

[TASK] Tugasmu adalah {deskripsi_tugas}.

[FORMAT] Berikan jawaban dalam format berikut:
- {poin_1}
- {poin_2}
- {poin_3}

[CONSTRAINTS]
- Gunakan bahasa yang {gaya_bahasa}
- Panjang jawaban: {panjang}
- Hindari: {hal_yang_dihindari}

[INPUT] {input_pengguna}
```
""",
        }
        return templates["default"] + f"\n> Template ini dapat disesuaikan untuk kebutuhan: **{input}**"

    except Exception as e:
        return f"Something went wrong with the generate_prompt_template tool: {e}"


def build_agent():
    load_dotenv()

    llm = Replicate(model="google/gemini-2.5-flash")

    system_message = """
Kamu adalah asisten AI yang ahli dalam Prompt Engineering untuk berbagai model AI (GPT, Claude, Gemini, Llama, dll.).
Tugasmu adalah membantu pengguna membuat prompt yang lengkap, efektif, dan terstruktur.

Kamu memiliki akses ke dua tool:
1. `prompt_technique_tip` — Menghasilkan tips atau teknik prompting secara acak.
2. `generate_prompt_template` — Menghasilkan template prompt lengkap berdasarkan use case yang diberikan.

Panduan penggunaan tool:
- Jika pengguna meminta "tips prompt", "teknik prompting", atau hal serupa → gunakan `prompt_technique_tip`.
- Jika pengguna meminta "buatkan prompt untuk...", "template prompt", atau "contoh prompt" → gunakan `generate_prompt_template` dengan use case sebagai input.
- Jika pertanyaan bersifat penjelasan atau teori → jawab sendiri menggunakan pengetahuanmu.

Keahlianmu meliputi:
- Struktur prompt: Role, Context, Task, Format, Constraints
- Teknik: Zero-shot, Few-shot, Chain-of-Thought, Tree-of-Thought, ReAct
- Optimasi prompt untuk berbagai model AI
- Prompt untuk kebutuhan spesifik: coding, penulisan, analisis data, creative writing, dll.

Gaya berbicara:
- Profesional namun mudah dipahami
- Berikan contoh konkret jika memungkinkan
- Ringkas dan langsung ke poin, kecuali diminta detail lebih

Mulai jawab sesuai peranmu sebagai asisten Prompt Engineering.
"""

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    tools = [
        prompt_technique_tip,
        generate_prompt_template,
    ]

    agent_executor = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={"system_message": system_message},
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )

    return agent_executor
