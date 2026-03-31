import streamlit as st
import os
import pickle
import numpy as np
import json
import base64
import re
import threading
import random
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ====================================
# CONFIG
# ====================================
st.set_page_config(
    page_title="UniAssist NCU",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================
# GEMINI SETUP
# ====================================
api_key = "AIzaSyDvu0dozL4pOqfS9QO7alR70QkVqX5kceY"  # paste your Gemini key here
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ====================================
# LOGO
# ====================================
def get_logo_base64():
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_base64()

# ====================================
# IMPORT VOICE MODULE
# ====================================
try:
    from Voice_module import get_voice_input, speak_response
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# ====================================
# GREETINGS
# ====================================
def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        time_greet = "Good Morning"
        emoji = "🌅"
    elif hour < 17:
        time_greet = "Good Afternoon"
        emoji = "☀️"
    else:
        time_greet = "Good Evening"
        emoji = "🌙"
    return time_greet, emoji

WELCOME_MESSAGES = [
    "Hello! I'm UniAssist, your AI guide to The NorthCap University. How can I help you today?",
    "Welcome to NCU! Ask me anything about admissions, courses, fees, placements, or campus life.",
    "Hi there! I'm here to help you explore everything about The NorthCap University. What would you like to know?",
]

SUGGESTED_QUESTIONS = [
    "What are the BTech admission requirements?",
    "Tell me about the fee structure",
    "What placements does NCU offer?",
    "Are there scholarships available?",
    "What hostel facilities are available?",
    "Tell me about the campus facilities",
]

# ====================================
# FULL CSS — Production Dark Theme
# ====================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #03070f;
    --bg2:       #070d1a;
    --bg3:       #0a1225;
    --card:      #0d1830;
    --gold:      #C5A572;
    --gold2:     #e2c98a;
    --gold-dim:  #6b5430;
    --blue:      #1a6fff;
    --blue-dim:  #0d3a8a;
    --text:      #dde4f0;
    --text2:     #7a8aaa;
    --text3:     #3a4a6a;
    --border:    rgba(197,165,114,0.15);
    --border2:   rgba(197,165,114,0.08);
    --glow:      rgba(197,165,114,0.12);
    --glow2:     rgba(26,111,255,0.12);
    --red:       #ff4d6d;
    --green:     #00d68f;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* Animated mesh background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 10% 20%, rgba(197,165,114,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(26,111,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(197,165,114,0.02) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
.block-container {
    padding: 1rem 1.5rem 2rem !important;
    max-width: 1000px !important;
    position: relative;
    z-index: 1;
}

/* ===== HEADER ===== */
.main-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 20px 28px;
    background: linear-gradient(135deg, rgba(13,24,48,0.97), rgba(7,13,26,0.97));
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 40px rgba(0,0,0,0.5), 0 0 0 1px var(--border2);
    animation: headerReveal 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--gold) 40%, var(--gold2) 60%, transparent 100%);
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(197,165,114,0.2) 50%, transparent 100%);
}
@keyframes headerReveal {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.header-logo img {
    height: 56px;
    filter: drop-shadow(0 0 14px rgba(197,165,114,0.35));
    animation: logoGlow 3s ease-in-out infinite alternate;
}
@keyframes logoGlow {
    from { filter: drop-shadow(0 0 8px rgba(197,165,114,0.2)); }
    to   { filter: drop-shadow(0 0 20px rgba(197,165,114,0.5)); }
}
.header-info { flex: 1; }
.header-title {
    font-family: 'Cinzel', serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 2px;
    text-shadow: 0 0 30px rgba(197,165,114,0.4);
    line-height: 1.2;
}
.header-sub {
    font-size: 12px;
    color: var(--text2);
    margin-top: 4px;
    letter-spacing: 0.5px;
}
.header-badges {
    display: flex;
    gap: 8px;
    align-items: center;
}
.badge {
    background: rgba(197,165,114,0.08);
    border: 1px solid rgba(197,165,114,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: var(--gold);
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.badge.online {
    background: rgba(0,214,143,0.08);
    border-color: rgba(0,214,143,0.25);
    color: var(--green);
    display: flex;
    align-items: center;
    gap: 5px;
}
.badge.online::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    animation: blink 1.5s infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ===== GREETING CARD ===== */
.greeting-card {
    background: linear-gradient(135deg, rgba(13,24,48,0.9), rgba(10,18,37,0.9));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) 0.2s both;
}
.greeting-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg at 50% 50%,
        transparent 0deg,
        rgba(197,165,114,0.03) 60deg,
        transparent 120deg,
        rgba(26,111,255,0.02) 180deg,
        transparent 240deg,
        rgba(197,165,114,0.03) 300deg,
        transparent 360deg);
    animation: rotate 20s linear infinite;
}
@keyframes rotate { to { transform: rotate(360deg); } }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.greeting-time {
    font-size: 13px;
    color: var(--gold);
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'Cinzel', serif;
    margin-bottom: 10px;
    position: relative;
}
.greeting-headline {
    font-family: 'Cinzel', serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
    position: relative;
    background: linear-gradient(135deg, var(--gold2), var(--text), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.greeting-msg {
    font-size: 14px;
    color: var(--text2);
    line-height: 1.7;
    max-width: 560px;
    margin: 0 auto 24px;
    position: relative;
}
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    position: relative;
}
.suggest-chip {
    background: rgba(197,165,114,0.06);
    border: 1px solid rgba(197,165,114,0.2);
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 12.5px;
    color: var(--gold);
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
}
.suggest-chip:hover {
    background: rgba(197,165,114,0.14);
    border-color: var(--gold);
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(197,165,114,0.15);
}

/* ===== CHAT MESSAGES ===== */
.chat-area {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-bottom: 12px;
}

.msg-row-user {
    display: flex;
    justify-content: flex-end;
    animation: slideRight 0.35s cubic-bezier(0.16,1,0.3,1) both;
    margin-bottom: 6px;
}
.msg-row-bot {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
    animation: slideLeft 0.35s cubic-bezier(0.16,1,0.3,1) both;
    margin-bottom: 6px;
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(24px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideLeft {
    from { opacity: 0; transform: translateX(-24px); }
    to   { opacity: 1; transform: translateX(0); }
}

.bot-avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--gold-dim), #2a1a08);
    border: 1.5px solid rgba(197,165,114,0.4);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 0 16px rgba(197,165,114,0.25);
    margin-top: 4px;
}

.bubble-user {
    background: linear-gradient(135deg, #162848, #0c1e38);
    border: 1px solid rgba(26,111,255,0.2);
    border-radius: 20px 20px 4px 20px;
    padding: 13px 18px;
    max-width: 68%;
    color: #c8d8f0;
    font-size: 14.5px;
    line-height: 1.65;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.bubble-bot {
    background: linear-gradient(135deg, #0d1830, #080f20);
    border: 1px solid var(--border);
    border-radius: 4px 20px 20px 20px;
    padding: 14px 18px;
    max-width: 78%;
    color: var(--text);
    font-size: 14.5px;
    line-height: 1.75;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    position: relative;
}
.bubble-bot::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, var(--gold-dim), transparent);
    border-radius: 4px 20px 0 0;
    opacity: 0.5;
}

/* ===== INPUT AREA ===== */
.stChatInputContainer {
    background: transparent !important;
}
.stChatInputContainer > div {
    background: rgba(13,24,48,0.95) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 0 1px var(--border2), 0 8px 32px rgba(0,0,0,0.4) !important;
    transition: border-color 0.3s !important;
}
.stChatInputContainer > div:focus-within {
    border-color: rgba(197,165,114,0.4) !important;
    box-shadow: 0 0 0 3px rgba(197,165,114,0.08), 0 8px 32px rgba(0,0,0,0.4) !important;
}
.stChatInputContainer textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

.sidebar-header {
    padding: 24px 16px 20px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
    background: linear-gradient(180deg, rgba(13,24,48,0.6), transparent);
}

.sb-section {
    font-family: 'Cinzel', serif;
    font-size: 10px;
    color: var(--gold-dim);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 16px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border2);
}

/* Voice Panel */
.voice-panel {
    background: linear-gradient(135deg, rgba(13,24,48,0.8), rgba(7,13,26,0.8));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 16px;
    margin-bottom: 12px;
    text-align: center;
}
.voice-orb-wrap {
    position: relative;
    width: 80px; height: 80px;
    margin: 0 auto 12px;
}
.voice-orb {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #162848, #040b18);
    border: 2px solid var(--gold-dim);
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    box-shadow: 0 0 24px rgba(197,165,114,0.15), inset 0 1px 0 rgba(197,165,114,0.1);
    transition: all 0.3s;
    position: relative;
    z-index: 2;
}
.voice-ring {
    position: absolute;
    inset: -8px;
    border-radius: 50%;
    border: 1px solid rgba(197,165,114,0.15);
    animation: ringPulse 2.5s ease-in-out infinite;
}
.voice-ring:nth-child(2) { inset: -16px; animation-delay: 0.5s; }
@keyframes ringPulse {
    0%,100% { opacity: 0.4; transform: scale(1); }
    50%      { opacity: 0.1; transform: scale(1.08); }
}

/* Waves */
.wave-row {
    display: flex; align-items: center; justify-content: center;
    gap: 3px; height: 28px; margin: 8px 0;
}
.wbar {
    width: 3px; border-radius: 3px;
    background: var(--gold); opacity: 0.2;
    transition: opacity 0.3s;
}
.wbar.active { opacity: 1; animation: wv 0.6s ease-in-out infinite alternate; }
.wbar:nth-child(1){height:6px;animation-delay:0s;}
.wbar:nth-child(2){height:14px;animation-delay:.08s;}
.wbar:nth-child(3){height:22px;animation-delay:.16s;}
.wbar:nth-child(4){height:18px;animation-delay:.12s;}
.wbar:nth-child(5){height:10px;animation-delay:.04s;}
.wbar:nth-child(6){height:20px;animation-delay:.2s;}
.wbar:nth-child(7){height:8px;animation-delay:.08s;}
@keyframes wv { from{transform:scaleY(0.3);} to{transform:scaleY(1);} }

.voice-status {
    font-size: 11.5px;
    color: var(--text2);
    margin-bottom: 10px;
    min-height: 16px;
}
.voice-last {
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 11.5px;
    color: var(--text2);
    font-style: italic;
    text-align: left;
    min-height: 30px;
    margin-bottom: 10px;
    word-break: break-word;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--card), var(--bg3)) !important;
    color: var(--gold) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #152240, var(--card)) !important;
    border-color: var(--gold) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(197,165,114,0.12) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Toggle */
.stToggle { color: var(--text) !important; }

/* Citation */
.citation-card {
    background: rgba(13,24,48,0.7);
    border: 1px solid var(--border2);
    border-left: 3px solid var(--gold-dim);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 13px;
    color: var(--text2);
    line-height: 1.6;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* Metric */
[data-testid="stMetric"] {
    background: rgba(13,24,48,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* Scrollbar */
* { scrollbar-width: thin; scrollbar-color: var(--gold-dim) transparent; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 4px; }

/* Info/success/error */
.stAlert { border-radius: 10px !important; border: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ====================================
# LOAD VECTOR DB
# ====================================
@st.cache_resource
def load_resources():
    try:
        with open("vector_db/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        embeddings = np.load("vector_db/embeddings.npy")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return chunks, embeddings, embed_model, True
    except:
        return [], np.array([]), None, False

chunks, embeddings, embed_model, db_loaded = load_resources()

# ====================================
# RETRIEVAL + GENERATION
# ====================================
def retrieve_context(query, top_k=80):
    if not db_loaded or embed_model is None:
        return []
    qe = embed_model.encode([query])
    sims = cosine_similarity(qe, embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_idx]

def generate_answer(query):
    selected = retrieve_context(query)
    context = "\n\n".join(selected)[:30000] if selected else "No university data loaded."
    prompt = f"""You are UniAssist NCU, the official AI assistant of The NorthCap University.

INSTRUCTIONS:
- Answer strictly using provided context.
- Do NOT hallucinate.
- Structure answer with headings and bullet points.
- Be concise and helpful.
- If information not available, say: "The information is not available in the official university data."

Context:
{context}

Question:
{query}"""
    try:
        response = model.generate_content(prompt)
        return response.text, selected[:3]
    except Exception as e:
        return f"Error generating response: {str(e)}", []

# ====================================
# ANALYTICS
# ====================================
def log_query(query):
    try:
        if not os.path.exists("analytics.json"):
            with open("analytics.json", "w") as f:
                json.dump([], f)
        with open("analytics.json", "r") as f:
            logs = json.load(f)
        logs.append({"query": query, "time": datetime.now().isoformat()})
        with open("analytics.json", "w") as f:
            json.dump(logs, f)
    except:
        pass

# ====================================
# CLEAN TEXT FOR SPEECH
# ====================================
def clean_for_speech(text):
    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    clean = re.sub(r'\*(.*?)\*', r'\1', clean)
    clean = re.sub(r'#{1,6}\s', '', clean)
    clean = clean.replace('\n', ' ').replace('•', '').strip()
    return clean[:1000]

# ====================================
# SESSION STATE
# ====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "citations" not in st.session_state:
    st.session_state.citations = []
if "voice_last" not in st.session_state:
    st.session_state.voice_last = ""
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "suggested" not in st.session_state:
    st.session_state.suggested = None

# ====================================
# HEADER
# ====================================
logo_tag = f'<img src="data:image/png;base64,{logo_b64}" />' if logo_b64 else '<span style="font-size:42px;">🎓</span>'
time_greet, time_emoji = get_greeting()

st.markdown(f"""
<div class="main-header">
    <div class="header-logo">{logo_tag}</div>
    <div class="header-info">
        <div class="header-title">UniAssist NCU</div>
        <div class="header-sub">The Official AI Assistant · The NorthCap University, Gurugram</div>
    </div>
    <div class="header-badges">
        <div class="badge online">Online</div>
        <div class="badge">AI Powered</div>
        <div class="badge">{time_emoji} {time_greet}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================
# SIDEBAR
# ====================================
with st.sidebar:
    # Logo
    if logo_b64:
        st.markdown(f"""
        <div class="sidebar-header">
            <img src="data:image/png;base64,{logo_b64}"
                 style="height:64px;filter:drop-shadow(0 0 12px rgba(197,165,114,0.4));" />
            <div style="font-family:'Cinzel',serif;font-size:13px;color:var(--gold);
                        margin-top:10px;letter-spacing:1.5px;">NCU ASSISTANT</div>
            <div style="font-size:11px;color:var(--text2);margin-top:3px;">Powered by Gemini AI</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- VOICE ASSISTANT ----
    st.markdown('<div class="sb-section">🎙️ Voice Assistant</div>', unsafe_allow_html=True)

    if VOICE_AVAILABLE:
        st.markdown("""
        <div class="voice-panel">
            <div class="voice-orb-wrap">
                <div class="voice-ring"></div>
                <div class="voice-ring"></div>
                <div class="voice-orb">🎙️</div>
            </div>
            <div class="wave-row">
                <div class="wbar" id="b1"></div>
                <div class="wbar" id="b2"></div>
                <div class="wbar" id="b3"></div>
                <div class="wbar" id="b4"></div>
                <div class="wbar" id="b5"></div>
                <div class="wbar" id="b6"></div>
                <div class="wbar" id="b7"></div>
            </div>
            <div class="voice-status">Click below to speak your question</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Speak", key="btn_speak", use_container_width=True):
                with st.spinner("🔴 Listening..."):
                    transcript = get_voice_input()
                if transcript:
                    st.session_state.voice_last = transcript
                    st.session_state.pending_voice = transcript
                    st.rerun()
                else:
                    st.warning("Couldn't hear. Try again.")

        with col2:
            tts_on = st.toggle("🔊 Speak", value=False, key="tts_toggle")

        if st.session_state.voice_last:
            st.markdown(f"""
            <div class="voice-last">🗣️ "{st.session_state.voice_last}"</div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:rgba(255,77,109,0.08);border:1px solid rgba(255,77,109,0.2);
                    border-radius:10px;padding:12px;font-size:12px;color:#ff6b6b;text-align:center;">
            ❌ Voice_module.py not found
        </div>
        """, unsafe_allow_html=True)
        st.code("pip install sounddevice scipy gtts playsound==1.2.2", language="bash")

    # ---- QUICK QUESTIONS ----
    st.markdown('<div class="sb-section">⚡ Quick Questions</div>', unsafe_allow_html=True)
    quick_map = {
        "🎓  Scholarships":   "What scholarships are available at NCU?",
        "💰  Fee Structure":   "Explain the fee structure at NCU.",
        "🏢  Placements":      "Tell me about placements at NCU.",
        "📚  Courses":         "What courses does NCU offer?",
        "🏨  Hostel":          "Tell me about hostel facilities at NCU.",
        "📅  Admissions":      "How do I apply for admission to NCU?",
        "🏛️  Campus":          "Tell me about campus facilities at NCU.",
        "👨‍🏫  Faculty":         "Tell me about the faculty at NCU.",
    }
    for label, q in quick_map.items():
        if st.button(label, key=f"q_{label}"):
            st.session_state.pending_voice = q
            st.rerun()

    # ---- ADMIN ----
    st.markdown('<div class="sb-section">⚙️ Admin</div>', unsafe_allow_html=True)
    if st.button("📊  Analytics", key="analytics"):
        if os.path.exists("analytics.json"):
            with open("analytics.json") as f:
                logs = json.load(f)
            st.metric("Total Queries", len(logs))
            if logs:
                st.caption(f"Last: {logs[-1].get('query','')[:40]}...")
        else:
            st.info("No queries yet.")

    if st.button("🗑️  Clear Chat", key="clear"):
        st.session_state.chat_history = []
        st.session_state.citations = []
        st.session_state.greeted = False
        st.rerun()

    st.markdown("""
    <div style="padding:16px 0 8px;text-align:center;font-size:11px;color:var(--text3);">
        UniAssist NCU v2.0 · Built with ❤️<br>
        <span style="color:var(--gold-dim);">The NorthCap University</span>
    </div>
    """, unsafe_allow_html=True)

# ====================================
# PROCESS QUERY
# ====================================
query = None

if "pending_voice" in st.session_state:
    query = st.session_state.pending_voice
    del st.session_state.pending_voice

if "suggested" in st.session_state and st.session_state.suggested:
    query = st.session_state.suggested
    st.session_state.suggested = None

user_input = st.chat_input("Ask anything about NCU...")
if user_input:
    query = user_input

if query:
    log_query(query)
    st.session_state.chat_history.append(("user", query))
    st.session_state.greeted = True

    with st.spinner("UniAssist is thinking..."):
        answer, citations = generate_answer(query)

    st.session_state.chat_history.append(("bot", answer))
    st.session_state.citations = citations

    if VOICE_AVAILABLE and st.session_state.get("tts_toggle", False):
        clean_answer = clean_for_speech(answer)
        t = threading.Thread(target=speak_response, args=(clean_answer,))
        t.daemon = True
        t.start()

    st.rerun()

# ====================================
# MAIN CONTENT AREA
# ====================================
if not st.session_state.chat_history:
    # GREETING CARD
    welcome_msg = random.choice(WELCOME_MESSAGES)
    st.markdown(f"""
    <div class="greeting-card">
        <div class="greeting-time">{time_emoji} {time_greet}</div>
        <div class="greeting-headline">Welcome to UniAssist NCU</div>
        <div class="greeting-msg">{welcome_msg}</div>
        <div class="suggestions">
            <div class="suggest-chip" onclick="void(0)">🎓 Admissions</div>
            <div class="suggest-chip">💰 Fee Structure</div>
            <div class="suggest-chip">🏢 Placements</div>
            <div class="suggest-chip">📚 Courses</div>
            <div class="suggest-chip">🏨 Hostel</div>
            <div class="suggest-chip">🎖️ Scholarships</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion buttons that actually work
    st.markdown("##### 💡 Try asking:")
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 3]:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.suggested = q
                st.rerun()

else:
    # CHAT MESSAGES
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
            <div class="msg-row-user">
                <div class="bubble-user">{message}</div>
            </div>""", unsafe_allow_html=True)
        else:
            fmt = message.replace('\n', '<br>')
            st.markdown(f"""
            <div class="msg-row-bot">
                <div class="bot-avatar">🤖</div>
                <div class="bubble-bot">{fmt}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # CITATIONS
    if st.session_state.citations:
        with st.expander("📚 Source References", expanded=False):
            for i, chunk in enumerate(st.session_state.citations):
                st.markdown(
                    f'<div class="citation-card"><strong style="color:var(--gold);">Source {i+1}</strong><br>{chunk[:350]}...</div>',
                    unsafe_allow_html=True)

    # NOT GREETED YET — show quick suggestions again below chat
    st.markdown("""
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border2);">
    """, unsafe_allow_html=True)
    cols = st.columns(3)
    quick_suggests = list(quick_map.items())[:6]
    for i, (label, q) in enumerate(quick_suggests):
        with cols[i % 3]:
            if st.button(label, key=f"chat_sug_{i}", use_container_width=True):
                st.session_state.pending_voice = q
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

