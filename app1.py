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
api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDyBG3EK3zboniV-yc3-wSBKtbIIhetqvE")
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
        return "Good Morning", "🌅"
    elif hour < 17:
        return "Good Afternoon", "☀️"
    else:
        return "Good Evening", "🌙"

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
    "Tell me about campus facilities",
]

# ====================================
# FULL CSS — Siri-level futuristic dark theme
# ====================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #03070f;
    --bg2:       #060c1a;
    --bg3:       #091220;
    --card:      #0b1628;
    --gold:      #C5A572;
    --gold2:     #e8d09a;
    --gold-dim:  #6b5430;
    --blue:      #4fc3f7;
    --blue2:     #0288d1;
    --accent:    #7c4dff;
    --text:      #dde4f0;
    --text2:     #7a8aaa;
    --text3:     #3a4a6a;
    --border:    rgba(197,165,114,0.18);
    --border2:   rgba(197,165,114,0.07);
    --green:     #00e5a0;
    --red:       #ff4d6d;
    --r:         16px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

/* Animated particle background */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 90% 60% at 10% 15%, rgba(197,165,114,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 70% 50% at 85% 80%, rgba(79,195,247,0.05) 0%, transparent 50%),
        radial-gradient(ellipse 50% 70% at 50% 40%, rgba(124,77,255,0.03) 0%, transparent 65%);
    pointer-events: none;
}
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
        radial-gradient(circle 1px at 20% 30%, rgba(197,165,114,0.3) 0%, transparent 1px),
        radial-gradient(circle 1px at 80% 20%, rgba(79,195,247,0.25) 0%, transparent 1px),
        radial-gradient(circle 1px at 60% 70%, rgba(197,165,114,0.2) 0%, transparent 1px),
        radial-gradient(circle 1px at 40% 85%, rgba(79,195,247,0.2) 0%, transparent 1px),
        radial-gradient(circle 1px at 90% 60%, rgba(197,165,114,0.15) 0%, transparent 1px);
    pointer-events: none;
}

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
.block-container {
    padding: 1rem 1.5rem 2rem !important;
    max-width: 1060px !important;
    position: relative; z-index: 1;
}

/* ===== HEADER ===== */
.main-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 18px 28px;
    background: linear-gradient(135deg, rgba(11,22,40,0.98), rgba(6,12,26,0.98));
    border: 1px solid var(--border);
    border-radius: 22px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 48px rgba(0,0,0,0.6), 0 0 0 1px var(--border2);
    animation: headerReveal 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent 0%, var(--gold-dim) 20%, var(--gold) 50%, var(--gold2) 65%, transparent 100%);
}
@keyframes headerReveal {
    from { opacity: 0; transform: translateY(-20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.header-logo img {
    height: 52px;
    filter: drop-shadow(0 0 16px rgba(197,165,114,0.4));
    animation: logoGlow 3s ease-in-out infinite alternate;
}
@keyframes logoGlow {
    from { filter: drop-shadow(0 0 8px rgba(197,165,114,0.2)); }
    to   { filter: drop-shadow(0 0 24px rgba(197,165,114,0.55)); }
}
.header-info { flex: 1; }
.header-title {
    font-family: 'Orbitron', monospace;
    font-size: 20px;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 3px;
    text-shadow: 0 0 30px rgba(197,165,114,0.35);
}
.header-sub {
    font-size: 11.5px;
    color: var(--text2);
    margin-top: 4px;
    letter-spacing: 0.3px;
}
.header-right { display: flex; align-items: center; gap: 10px; }
.badge {
    background: rgba(197,165,114,0.07);
    border: 1px solid rgba(197,165,114,0.18);
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 11px;
    color: var(--gold);
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.badge.online {
    background: rgba(0,229,160,0.07);
    border-color: rgba(0,229,160,0.25);
    color: var(--green);
    display: flex; align-items: center; gap: 5px;
}
.badge.online::before {
    content: '';
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    animation: blink 1.5s infinite;
}
@keyframes blink { 0%,100%{opacity:1;box-shadow:0 0 6px var(--green);} 50%{opacity:0.3;box-shadow:none;} }

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    animation: fadeUp 0.5s ease 0.15s both;
}
.stat-item {
    flex: 1;
    background: linear-gradient(135deg, rgba(11,22,40,0.9), rgba(6,12,26,0.9));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}
.stat-item::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(197,165,114,0.3), transparent);
}
.stat-item:hover {
    border-color: rgba(197,165,114,0.35);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.stat-icon { font-size: 22px; }
.stat-val {
    font-family: 'Orbitron', monospace;
    font-size: 18px;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.stat-label { font-size: 11px; color: var(--text2); margin-top: 2px; }

/* ===== GREETING CARD ===== */
.greeting-card {
    background: linear-gradient(135deg, rgba(11,22,40,0.95), rgba(9,18,32,0.95));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 36px 40px;
    margin-bottom: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) 0.2s both;
}
.greeting-card::before {
    content: '';
    position: absolute;
    top: -80%; left: -80%;
    width: 260%; height: 260%;
    background: conic-gradient(from 0deg at 50% 50%,
        transparent 0deg, rgba(197,165,114,0.04) 60deg,
        transparent 120deg, rgba(79,195,247,0.03) 180deg,
        transparent 240deg, rgba(197,165,114,0.04) 300deg, transparent 360deg);
    animation: rotate 25s linear infinite;
}
@keyframes rotate { to { transform: rotate(360deg); } }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* AI Avatar in greeting */
.ai-avatar-hero {
    width: 90px; height: 90px;
    border-radius: 50%;
    margin: 0 auto 20px;
    position: relative;
    display: flex; align-items: center; justify-content: center;
}
.ai-avatar-hero .orb {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #1a3060, #060e20);
    border: 2px solid rgba(197,165,114,0.5);
    display: flex; align-items: center; justify-content: center;
    font-size: 36px;
    box-shadow: 0 0 30px rgba(197,165,114,0.2), inset 0 1px 0 rgba(197,165,114,0.15);
    animation: orbFloat 4s ease-in-out infinite;
    position: relative; z-index: 2;
}
@keyframes orbFloat {
    0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);}
}
.ai-avatar-hero .ring {
    position: absolute; border-radius: 50%;
    border: 1px solid rgba(197,165,114,0.2);
    animation: ringExpand 2.5s ease-in-out infinite;
}
.ai-avatar-hero .ring:nth-child(1){inset:-10px;}
.ai-avatar-hero .ring:nth-child(2){inset:-22px;animation-delay:0.5s;}
.ai-avatar-hero .ring:nth-child(3){inset:-36px;animation-delay:1s;}
@keyframes ringExpand {
    0%{opacity:0.5;transform:scale(1);}
    100%{opacity:0;transform:scale(1.15);}
}

.greeting-time {
    font-size: 11px; color: var(--gold);
    letter-spacing: 4px; text-transform: uppercase;
    font-family: 'Orbitron', monospace;
    margin-bottom: 8px; position: relative;
}
.greeting-headline {
    font-family: 'Orbitron', monospace;
    font-size: 26px; font-weight: 700;
    margin-bottom: 10px; position: relative;
    background: linear-gradient(135deg, var(--gold2), var(--text), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.greeting-msg {
    font-size: 14px; color: var(--text2);
    line-height: 1.75; max-width: 540px;
    margin: 0 auto 28px; position: relative;
}
.suggestions {
    display: flex; flex-wrap: wrap;
    gap: 8px; justify-content: center; position: relative;
}
.suggest-chip {
    background: rgba(197,165,114,0.06);
    border: 1px solid rgba(197,165,114,0.2);
    border-radius: 24px;
    padding: 8px 18px;
    font-size: 12.5px; color: var(--gold);
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1);
}
.suggest-chip:hover {
    background: rgba(197,165,114,0.14);
    border-color: var(--gold);
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 6px 20px rgba(197,165,114,0.2);
}

/* ===== CHAT MESSAGES ===== */
.chat-area { display: flex; flex-direction: column; gap: 6px; padding-bottom: 12px; }

.msg-row-user {
    display: flex; justify-content: flex-end;
    animation: slideRight 0.35s cubic-bezier(0.16,1,0.3,1) both;
    margin-bottom: 4px;
}
.msg-row-bot {
    display: flex; justify-content: flex-start;
    align-items: flex-start; gap: 12px;
    animation: slideLeft 0.35s cubic-bezier(0.16,1,0.3,1) both;
    margin-bottom: 4px;
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(28px) scale(0.96); }
    to   { opacity: 1; transform: translateX(0) scale(1); }
}
@keyframes slideLeft {
    from { opacity: 0; transform: translateX(-28px) scale(0.96); }
    to   { opacity: 1; transform: translateX(0) scale(1); }
}

.bot-avatar {
    width: 40px; height: 40px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #1a3060, #040b18);
    border: 1.5px solid rgba(197,165,114,0.4);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    box-shadow: 0 0 18px rgba(197,165,114,0.2);
    margin-top: 4px;
    animation: avatarGlow 3s ease-in-out infinite alternate;
}
@keyframes avatarGlow {
    from{box-shadow:0 0 10px rgba(197,165,114,0.15);}
    to{box-shadow:0 0 24px rgba(197,165,114,0.35);}
}

.bubble-user {
    background: linear-gradient(135deg, #162848, #0c1e38);
    border: 1px solid rgba(79,195,247,0.18);
    border-radius: 20px 20px 4px 20px;
    padding: 13px 18px;
    max-width: 66%;
    color: #c8d8f0;
    font-size: 14.5px; line-height: 1.65;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}

.bubble-bot {
    background: linear-gradient(135deg, #0b1628, #060e20);
    border: 1px solid var(--border);
    border-radius: 4px 20px 20px 20px;
    padding: 15px 20px;
    max-width: 78%;
    color: var(--text);
    font-size: 14.5px; line-height: 1.8;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    position: relative;
}
.bubble-bot::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, var(--gold-dim), rgba(197,165,114,0.1), transparent);
    border-radius: 4px 20px 0 0;
}
.bot-label {
    font-family: 'Orbitron', monospace;
    font-size: 9px; color: var(--gold);
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 6px; opacity: 0.7;
}

/* Typing indicator */
.typing-indicator {
    display: flex; align-items: center; gap: 10px;
    animation: slideLeft 0.35s ease both;
    margin-bottom: 6px;
}
.typing-bubble {
    background: linear-gradient(135deg, #0b1628, #060e20);
    border: 1px solid var(--border);
    border-radius: 4px 20px 20px 20px;
    padding: 14px 18px;
    display: flex; align-items: center; gap: 5px;
}
.typing-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--gold);
    animation: typingBounce 1.2s ease-in-out infinite;
}
.typing-dot:nth-child(2){animation-delay:0.2s;}
.typing-dot:nth-child(3){animation-delay:0.4s;}
@keyframes typingBounce {
    0%,60%,100%{transform:translateY(0);opacity:0.4;}
    30%{transform:translateY(-8px);opacity:1;}
}

/* ===== INPUT BOX — NUCLEAR DARK FIX ===== */
/* Kill ALL white backgrounds everywhere */
body, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stBottom"],
[data-testid="stBottom"] > *,
[data-testid="stBottom"] > * > *,
[data-testid="stBottom"] > * > * > *,
.stChatInputContainer,
.stChatInputContainer *,
[data-testid="stChatInput"],
[data-testid="stChatInput"] *,
div[class*="chatInput"],
div[class*="chatInput"] *,
div[class*="InputContainer"],
div[class*="InputContainer"] * {
    background-color: transparent !important;
}

/* Force bottom area dark */
[data-testid="stBottom"] {
    background: #03070f !important;
    background-color: #03070f !important;
    border-top: 1px solid rgba(197,165,114,0.1) !important;
    padding: 12px 0 0 !important;
    position: relative !important;
}
[data-testid="stBottom"]::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: #03070f !important;
    z-index: -1 !important;
}

/* The actual input box */
.stChatInputContainer > div,
[data-testid="stChatInput"] > div {
    background: #0b1628 !important;
    background-color: #0b1628 !important;
    border: 1.5px solid rgba(197,165,114,0.3) !important;
    border-radius: 18px !important;
    box-shadow:
        0 0 0 1px rgba(197,165,114,0.06),
        0 0 30px rgba(197,165,114,0.05),
        inset 0 1px 0 rgba(197,165,114,0.05) !important;
    transition: all 0.3s !important;
}
.stChatInputContainer > div:focus-within,
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(197,165,114,0.6) !important;
    box-shadow:
        0 0 0 4px rgba(197,165,114,0.08),
        0 0 40px rgba(197,165,114,0.08),
        inset 0 1px 0 rgba(197,165,114,0.1) !important;
}

/* Textarea — visible text when typing */
.stChatInputContainer textarea,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"],
[data-testid="stChatInputTextArea"] * {
    background: transparent !important;
    background-color: transparent !important;
    color: #e8d09a !important;
    caret-color: #C5A572 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
    font-weight: 400 !important;
    -webkit-text-fill-color: #e8d09a !important;
}
.stChatInputContainer textarea::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
    color: #3a4a6a !important;
    -webkit-text-fill-color: #3a4a6a !important;
}

/* Send button — gold glowing */
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #6b5430, #3d2a10) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(197,165,114,0.4) !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: linear-gradient(135deg, #C5A572, #8a6d3c) !important;
    box-shadow: 0 0 20px rgba(197,165,114,0.4), 0 4px 16px rgba(0,0,0,0.4) !important;
    transform: scale(1.1) rotate(5deg) !important;
}
[data-testid="stChatInputSubmitButton"] button svg {
    fill: #e8d09a !important;
    color: #e8d09a !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.sidebar-header {
    padding: 24px 16px 20px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
    background: linear-gradient(180deg, rgba(11,22,40,0.7), transparent);
}

.sb-section {
    font-family: 'Orbitron', monospace;
    font-size: 9px; color: var(--gold-dim);
    letter-spacing: 3px; text-transform: uppercase;
    margin: 16px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border2);
}

/* ===== SIRI-STYLE VOICE PANEL ===== */
.voice-panel {
    background: linear-gradient(145deg, rgba(11,22,40,0.95), rgba(6,12,26,0.95));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px 16px;
    margin-bottom: 14px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.voice-panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}

/* Siri orb */
.siri-orb-wrap {
    width: 100px; height: 100px;
    margin: 0 auto 14px;
    position: relative;
    display: flex; align-items: center; justify-content: center;
}
.siri-orb {
    width: 84px; height: 84px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #1e3a6e, #050c1e);
    border: 2px solid rgba(197,165,114,0.45);
    display: flex; align-items: center; justify-content: center;
    font-size: 34px;
    box-shadow:
        0 0 30px rgba(197,165,114,0.2),
        0 0 60px rgba(197,165,114,0.08),
        inset 0 1px 0 rgba(197,165,114,0.12);
    position: relative; z-index: 3;
    transition: all 0.4s cubic-bezier(0.34,1.56,0.64,1);
    cursor: pointer;
    animation: siriFloat 4s ease-in-out infinite;
}
.siri-orb:hover {
    transform: scale(1.06);
    box-shadow: 0 0 40px rgba(197,165,114,0.35), 0 0 80px rgba(197,165,114,0.12);
    border-color: var(--gold);
}
@keyframes siriFloat {
    0%,100%{transform:translateY(0);}
    50%{transform:translateY(-5px);}
}
.siri-ring {
    position: absolute; border-radius: 50%;
    border: 1.5px solid rgba(197,165,114,0.15);
    animation: siriRingPulse 2.5s ease-in-out infinite;
}
.siri-ring:nth-child(1){inset:-8px;animation-delay:0s;}
.siri-ring:nth-child(2){inset:-18px;animation-delay:0.5s;}
.siri-ring:nth-child(3){inset:-30px;animation-delay:1s;border-color:rgba(79,195,247,0.08);}
@keyframes siriRingPulse {
    0%{opacity:0.6;transform:scale(1);}
    50%{opacity:0.15;transform:scale(1.06);}
    100%{opacity:0.6;transform:scale(1);}
}

/* Listening state */
.siri-orb.listening {
    border-color: var(--green) !important;
    box-shadow: 0 0 40px rgba(0,229,160,0.35), 0 0 80px rgba(0,229,160,0.12) !important;
    animation: siriListen 0.8s ease-in-out infinite alternate !important;
}
@keyframes siriListen {
    from{transform:scale(1);}
    to{transform:scale(1.08);}
}

/* Speaking state */
.siri-orb.speaking {
    border-color: var(--blue) !important;
    box-shadow: 0 0 40px rgba(79,195,247,0.35), 0 0 80px rgba(79,195,247,0.12) !important;
}

/* Siri equalizer */
.siri-eq {
    display: flex; align-items: center;
    justify-content: center; gap: 3px;
    height: 36px; margin: 6px 0;
}
.siri-bar {
    border-radius: 3px;
    background: linear-gradient(180deg, var(--gold2), var(--gold-dim));
    opacity: 0.2;
    transition: all 0.3s;
}
.siri-bar.active {
    opacity: 1;
    animation: siEq 0.5s ease-in-out infinite alternate;
}
.siri-bar:nth-child(1){width:3px;height:8px;animation-delay:0s;}
.siri-bar:nth-child(2){width:3px;height:16px;animation-delay:0.07s;}
.siri-bar:nth-child(3){width:3px;height:26px;animation-delay:0.14s;}
.siri-bar:nth-child(4){width:3px;height:22px;animation-delay:0.1s;}
.siri-bar:nth-child(5){width:4px;height:30px;animation-delay:0.05s;}
.siri-bar:nth-child(6){width:3px;height:22px;animation-delay:0.12s;}
.siri-bar:nth-child(7){width:3px;height:26px;animation-delay:0.18s;}
.siri-bar:nth-child(8){width:3px;height:16px;animation-delay:0.08s;}
.siri-bar:nth-child(9){width:3px;height:8px;animation-delay:0.03s;}
@keyframes siEq {
    from{transform:scaleY(0.25);}
    to{transform:scaleY(1);}
}

/* Siri wave gradient (decorative) */
.siri-wave {
    width: 100%; height: 24px;
    position: relative; margin: 4px 0;
    overflow: hidden;
}
.siri-wave::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(90deg,
        transparent 0%, rgba(197,165,114,0.3) 20%,
        rgba(79,195,247,0.4) 50%, rgba(197,165,114,0.3) 80%,
        transparent 100%);
    animation: waveSweep 2s linear infinite;
    height: 2px; top: 50%; border-radius: 2px;
    opacity: 0.4;
}
@keyframes waveSweep { from{transform:translateX(-100%);} to{transform:translateX(100%);} }

.voice-status {
    font-size: 11.5px; color: var(--text2);
    margin-bottom: 10px; min-height: 16px;
    transition: color 0.3s;
}
.voice-status.active { color: var(--green); }
.voice-status.speaking { color: var(--blue); }

.voice-transcript {
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px; color: var(--text2);
    font-style: italic; text-align: left;
    min-height: 32px; margin-bottom: 10px;
    word-break: break-word;
    transition: all 0.3s;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, rgba(11,22,40,0.95), rgba(6,12,26,0.95)) !important;
    color: var(--gold) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, rgba(197,165,114,0.08), transparent) !important;
    opacity: 0 !important;
    transition: opacity 0.25s !important;
}
.stButton > button:hover {
    border-color: rgba(197,165,114,0.45) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 6px 24px rgba(197,165,114,0.15), 0 0 0 1px rgba(197,165,114,0.12) !important;
    color: var(--gold2) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

/* Toggle */
.stToggle label { color: var(--text2) !important; font-size: 12.5px !important; }

/* Citation */
.citation-card {
    background: rgba(11,22,40,0.7);
    border: 1px solid var(--border2);
    border-left: 3px solid var(--gold-dim);
    border-radius: 10px;
    padding: 12px 16px; margin-bottom: 10px;
    font-size: 12.5px; color: var(--text2); line-height: 1.6;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* Metric */
[data-testid="stMetric"] {
    background: rgba(11,22,40,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* Scrollbar */
* { scrollbar-width: thin; scrollbar-color: var(--gold-dim) transparent; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 4px; }

/* Alert */
.stAlert { border-radius: 12px !important; border: 1px solid var(--border) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(11,22,40,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text2) !important;
}

/* Quick suggest section */
.suggest-section-title {
    font-family: 'Orbitron', monospace;
    font-size: 10px; color: var(--gold-dim);
    letter-spacing: 3px; text-transform: uppercase;
    margin: 14px 0 10px;
    display: flex; align-items: center; gap: 8px;
}
.suggest-section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--gold-dim), transparent);
}
</style>

<script>
// Force dark on every element Streamlit renders dynamically
function forceDark() {
    const selectors = [
        '[data-testid="stBottom"]',
        '[data-testid="stBottom"] > div',
        '[data-testid="stBottom"] > div > div',
        '[data-testid="stBottom"] > div > div > div',
        '.stChatInputContainer',
        '.stChatInputContainer > div',
        '[data-testid="stChatInput"]',
        '[data-testid="stChatInputTextArea"]',
    ];
    selectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => {
            el.style.setProperty('background', '#03070f', 'important');
            el.style.setProperty('background-color', '#03070f', 'important');
        });
    });

    // Fix textarea text color
    document.querySelectorAll('textarea').forEach(el => {
        el.style.setProperty('color', '#e8d09a', 'important');
        el.style.setProperty('-webkit-text-fill-color', '#e8d09a', 'important');
        el.style.setProperty('background', 'transparent', 'important');
    });
}

// Run immediately and keep watching for Streamlit re-renders
forceDark();
setInterval(forceDark, 300);
const obs = new MutationObserver(forceDark);
obs.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Extra injection after page loads
st.markdown("""
<style>
/* ABSOLUTE NUCLEAR — kill white bottom */
html { background: #03070f !important; }
body { background: #03070f !important; }

[data-testid="stBottom"],
[data-testid="stBottom"] *:not(textarea):not(button):not(svg):not(path) {
    background: #03070f !important;
    background-color: #03070f !important;
}

/* Input box itself styled nicely */
.stChatInputContainer > div {
    background: linear-gradient(135deg, #0d1e38, #081428) !important;
    border: 1.5px solid rgba(197,165,114,0.35) !important;
    border-radius: 16px !important;
    box-shadow:
        0 0 20px rgba(197,165,114,0.06),
        0 0 0 1px rgba(197,165,114,0.04),
        inset 0 1px 0 rgba(197,165,114,0.08) !important;
}
.stChatInputContainer > div:focus-within {
    border-color: rgba(197,165,114,0.7) !important;
    box-shadow:
        0 0 0 4px rgba(197,165,114,0.1),
        0 0 30px rgba(197,165,114,0.12),
        inset 0 1px 0 rgba(197,165,114,0.15) !important;
}

/* Textarea visible gold text */
textarea {
    color: #e8d09a !important;
    -webkit-text-fill-color: #e8d09a !important;
    background: transparent !important;
    background-color: transparent !important;
    caret-color: #C5A572 !important;
    font-size: 15px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
textarea::placeholder {
    color: #2a3a5a !important;
    -webkit-text-fill-color: #2a3a5a !important;
}
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
    import re as _r
    # Remove bold **text**
    clean = _r.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove italic *text*
    clean = _r.sub(r'\*(.*?)\*', r'\1', clean)
    # Remove headings ### ## #
    clean = _r.sub(r'#{1,6}\s*', '', clean)
    # Remove bullet dashes and stars at line start
    clean = _r.sub(r'(?m)^[\*\-\•]\s*', '', clean)
    # Remove remaining lone asterisks
    clean = clean.replace('*', '').replace('#', '').replace('◆', '')
    # Remove markdown links [text](url)
    clean = _r.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
    # Remove backticks
    clean = clean.replace('`', '')
    # Clean up whitespace
    clean = _r.sub(r'\n+', ' ', clean)
    clean = _r.sub(r'\s+', ' ', clean)
    return clean.strip()[:1000]

# ====================================
# SESSION STATE
# ====================================
for key, default in [
    ("chat_history", []),
    ("citations", []),
    ("voice_last", ""),
    ("greeted", False),
    ("suggested", None),
    ("voice_state", "idle"),
    ("voice_triggered", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ====================================
# HEADER
# ====================================
logo_tag = f'<img src="data:image/png;base64,{logo_b64}" />' if logo_b64 else '<span style="font-size:44px;">🎓</span>'
time_greet, time_emoji = get_greeting()

st.markdown(f"""
<div class="main-header">
    <div class="header-logo">{logo_tag}</div>
    <div class="header-info">
        <div class="header-title">UNIASSIST NCU</div>
        <div class="header-sub">🤖 AI Assistant · The NorthCap University, Gurugram</div>
    </div>
    <div class="header-right">
        <div class="badge online">Online</div>
        <div class="badge">✦ AI Powered</div>
        <div class="badge">{time_emoji} {time_greet}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================
# STATS BAR
# ====================================
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-icon">🏢</div>
        <div>
            <div class="stat-val">90%+</div>
            <div class="stat-label">Placements</div>
        </div>
    </div>
    <div class="stat-item">
        <div class="stat-icon">📚</div>
        <div>
            <div class="stat-val">78</div>
            <div class="stat-label">Courses</div>
        </div>
    </div>
    <div class="stat-item">
        <div class="stat-icon">🎓</div>
        <div>
            <div class="stat-val">15K+</div>
            <div class="stat-label">Alumni</div>
        </div>
    </div>
    <div class="stat-item">
        <div class="stat-icon">🏆</div>
        <div>
            <div class="stat-val">NAAC A</div>
            <div class="stat-label">Accredited</div>
        </div>
    </div>
    <div class="stat-item">
        <div class="stat-icon">🤝</div>
        <div>
            <div class="stat-val">250+</div>
            <div class="stat-label">Recruiters</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================
# SIDEBAR
# ====================================
with st.sidebar:
    # Logo / brand
    if logo_b64:
        st.markdown(f"""
        <div class="sidebar-header">
            <img src="data:image/png;base64,{logo_b64}"
                 style="height:62px;filter:drop-shadow(0 0 14px rgba(197,165,114,0.45));" />
            <div style="font-family:'Orbitron',monospace;font-size:12px;color:var(--gold);
                        margin-top:10px;letter-spacing:2px;">NCU ASSISTANT</div>
            <div style="font-size:11px;color:var(--text3);margin-top:3px;">The NorthCap University</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- SIRI-STYLE VOICE ASSISTANT ----
    st.markdown('<div class="sb-section">🎙 Voice Assistant</div>', unsafe_allow_html=True)

    vs = st.session_state.voice_state
    orb_class = "siri-orb"
    bar_class = ""
    status_text = "Tap below to speak your question"
    status_class = ""
    if vs == "listening":
        orb_class += " listening"
        bar_class = "active"
        status_text = "🔴 Listening... speak now"
        status_class = "active"
    elif vs == "speaking":
        orb_class += " speaking"
        bar_class = "active"
        status_text = "🔊 Speaking response..."
        status_class = "speaking"

    # Build equalizer bars
    bars = "".join([f'<div class="siri-bar {bar_class}"></div>' for _ in range(9)])

    transcript_html = f'🗣️ &ldquo;{st.session_state.voice_last}&rdquo;' if st.session_state.voice_last else '<span style="color:var(--text3)">No recent transcript</span>'

    st.markdown(f"""
    <div class="voice-panel">
        <div class="siri-orb-wrap">
            <div class="siri-ring"></div>
            <div class="siri-ring"></div>
            <div class="siri-ring"></div>
            <div class="{orb_class}">🎙️</div>
        </div>
        <div class="siri-eq">{bars}</div>
        <div class="siri-wave"></div>
        <div class="voice-status {status_class}">{status_text}</div>
        <div class="voice-transcript">{transcript_html}</div>
    </div>
    """, unsafe_allow_html=True)

    if VOICE_AVAILABLE:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Speak", key="btn_speak", use_container_width=True):
                st.session_state.voice_state = "listening"
                st.rerun()
        with col2:
            tts_on = st.toggle("🔊 Voice", value=True, key="tts_toggle")

        # Stop Speech button — only show when voice is ON
        if st.session_state.get("tts_toggle", True):
            if st.button("⏹️ Stop Speech", key="btn_stop", use_container_width=True):
                try:
                    import subprocess
                    # Kill any running pyttsx3 / speech process on Windows
                    subprocess.run(
                        ["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq pyttsx3*"],
                        capture_output=True
                    )
                    # More reliable: kill by finding our speech subprocess
                    import psutil
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            cmdline = ' '.join(proc.info['cmdline'] or [])
                            if 'pyttsx3' in cmdline and proc.pid != os.getpid():
                                proc.kill()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Stop error: {e}")
                st.success("Speech stopped!")

            st.markdown("""
            <div style="text-align:center;font-size:11px;color:var(--green);
                        margin-top:4px;letter-spacing:1px;">
                ● SPEECH TO SPEECH ACTIVE
            </div>""", unsafe_allow_html=True)

        # Process voice if in listening state
        if st.session_state.voice_state == "listening":
            with st.spinner("🔴 Listening... speak now"):
                transcript = get_voice_input()
            if transcript:
                st.session_state.voice_last = transcript
                st.session_state.pending_voice = transcript
                st.session_state.voice_triggered = True
                st.session_state.voice_state = "idle"
                st.rerun()
            else:
                st.session_state.voice_state = "idle"
                st.warning("Couldn't hear clearly. Try again.")
                st.rerun()
    else:
        st.markdown("""
        <div style="background:rgba(255,77,109,0.07);border:1px solid rgba(255,77,109,0.2);
                    border-radius:12px;padding:14px;font-size:12px;color:#ff7a8a;text-align:center;margin-bottom:8px;">
            ❌ Voice_module.py not found<br>
            <span style="color:var(--text3);font-size:11px;">Install dependencies below</span>
        </div>
        """, unsafe_allow_html=True)
        st.code("pip install sounddevice scipy gtts playsound==1.2.2", language="bash")
        st.caption("Then create Voice_module.py with get_voice_input() and speak_response()")

    # ---- QUICK QUESTIONS ----
    st.markdown('<div class="sb-section">⚡ Quick Questions</div>', unsafe_allow_html=True)
    quick_map = {
        "🎓 Scholarships":   "What scholarships are available at NCU?",
        "💰 Fee Structure":   "Explain the fee structure at NCU.",
        "🏢 Placements":      "Tell me about placements at NCU.",
        "📚 Courses":         "What courses does NCU offer?",
        "🏨 Hostel":          "Tell me about hostel facilities at NCU.",
        "📅 Admissions":      "How do I apply for admission to NCU?",
        "🏛️ Campus":          "Tell me about campus facilities at NCU.",
        "👨‍🏫 Faculty":         "Tell me about the faculty at NCU.",
    }
    for label, q in quick_map.items():
        if st.button(label, key=f"q_{label}"):
            st.session_state.pending_voice = q
            st.rerun()

    # ---- ADMIN ----
    st.markdown('<div class="sb-section">⚙️ Admin</div>', unsafe_allow_html=True)
    if st.button("📊 Analytics", key="analytics"):
        if os.path.exists("analytics.json"):
            with open("analytics.json") as f:
                logs = json.load(f)
            st.metric("Total Queries", len(logs))
            if logs:
                st.caption(f"Last: {logs[-1].get('query','')[:40]}...")
        else:
            st.info("No queries yet.")

    if st.button("🗑️ Clear Chat", key="clear"):
        st.session_state.chat_history = []
        st.session_state.citations = []
        st.session_state.greeted = False
        st.session_state.voice_state = "idle"
        st.rerun()

    st.markdown("""
    <div style="padding:18px 0 8px;text-align:center;font-size:11px;color:var(--text3);
                border-top:1px solid var(--border2);margin-top:12px;">
        UniAssist NCU v3.0 · Built with ❤️<br>
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

if st.session_state.get("suggested"):
    query = st.session_state.suggested
    st.session_state.suggested = None

user_input = st.chat_input("💬  Ask anything about NCU — admissions, fees, placements...")
if user_input:
    query = user_input

if query:
    log_query(query)
    st.session_state.chat_history.append(("user", query))
    st.session_state.greeted = True

    # ── Show typing indicator while generating ──
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="msg-row-bot" style="margin-bottom:8px;">
        <div class="bot-avatar">🤖</div>
        <div class="typing-bubble">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Generate answer ──
    answer, citations = generate_answer(query)
    typing_placeholder.empty()

    # ── Typewriter effect — stream word by word ──
    import re as _re

    def render_bot_message(text):
        fmt = text
        fmt = _re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:var(--gold2);">\1</strong>', fmt)
        fmt = _re.sub(r'\*(.*?)\*', r'<em>\1</em>', fmt)
        fmt = _re.sub(r'###\s+(.*?)(\n|$)', r'<div style="font-family:Orbitron,monospace;font-size:13px;color:var(--gold);margin:14px 0 6px;letter-spacing:1px;">\1</div>', fmt)
        fmt = _re.sub(r'##\s+(.*?)(\n|$)', r'<div style="font-family:Orbitron,monospace;font-size:14px;color:var(--gold);margin:14px 0 6px;letter-spacing:1px;">\1</div>', fmt)
        fmt = _re.sub(r'#\s+(.*?)(\n|$)', r'<div style="font-family:Orbitron,monospace;font-size:15px;color:var(--gold);margin:14px 0 8px;letter-spacing:1px;">\1</div>', fmt)
        fmt = _re.sub(r'(?m)^[\*\-]\s+(.+)$', r'<div style="display:flex;gap:8px;margin:4px 0;"><span style="color:var(--gold);margin-top:2px;">◆</span><span>\1</span></div>', fmt)
        fmt = fmt.replace('\n\n', '<div style="height:10px;"></div>')
        fmt = fmt.replace('\n', '<br>')
        return f"""
        <div class="msg-row-bot">
            <div class="bot-avatar">🤖</div>
            <div class="bubble-bot">
                <div class="bot-label">UniAssist · NCU</div>
                {fmt}
            </div>
        </div>"""

    # Stream words one by one
    stream_placeholder = st.empty()
    words = answer.split(" ")
    streamed = ""
    for i, word in enumerate(words):
        streamed += word + " "
        # Update every 3 words for smooth effect
        if i % 3 == 0 or i == len(words) - 1:
            stream_placeholder.markdown(
                render_bot_message(streamed + "▌"),
                unsafe_allow_html=True
            )
    # Final render without cursor
    stream_placeholder.empty()

    st.session_state.chat_history.append(("bot", answer))
    st.session_state.citations = citations

    # ── SPEAK for BOTH typing and voice (if Voice toggle is ON) ──
    if st.session_state.get("tts_toggle", True):
        clean_answer = clean_for_speech(answer)
        try:
            import subprocess, sys
            script = (
                f"import pyttsx3; "
                f"e=pyttsx3.init(); "
                f"e.setProperty('rate',155); "
                f"e.setProperty('volume',1.0); "
                f"e.say({repr(clean_answer)}); "
                f"e.runAndWait()"
            )
            subprocess.Popen(
                [sys.executable, "-c", script],
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        except Exception as e:
            print(f"TTS error: {e}")

    st.session_state.voice_triggered = False
    st.rerun()

# ====================================
# MAIN CONTENT AREA
# ====================================
if not st.session_state.chat_history:
    # GREETING CARD
    welcome_msg = random.choice(WELCOME_MESSAGES)
    st.markdown(f"""
    <div class="greeting-card">
        <div class="ai-avatar-hero">
            <div class="ring"></div>
            <div class="ring"></div>
            <div class="ring"></div>
            <div class="orb">🤖</div>
        </div>
        <div class="greeting-time">{time_emoji} {time_greet}</div>
        <div class="greeting-headline">Welcome to UniAssist NCU</div>
        <div class="greeting-msg">{welcome_msg}</div>
        <div class="suggestions">
            <div class="suggest-chip">🎓 Admissions</div>
            <div class="suggest-chip">💰 Fee Structure</div>
            <div class="suggest-chip">🏢 Placements</div>
            <div class="suggest-chip">📚 Courses</div>
            <div class="suggest-chip">🏨 Hostel</div>
            <div class="suggest-chip">🎖️ Scholarships</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="suggest-section-title">💡 Try asking</div>', unsafe_allow_html=True)
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
            # Convert markdown to clean HTML for proper rendering
            import re as _re
            fmt = message
            # Bold **text**
            fmt = _re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:var(--gold2);">\1</strong>', fmt)
            # Italic *text*
            fmt = _re.sub(r'\*(.*?)\*', r'<em>\1</em>', fmt)
            # Headings ## text
            fmt = _re.sub(r'###\s+(.*?)(\n|$)', r'<div style="font-family:Orbitron,monospace;font-size:13px;color:var(--gold);margin:14px 0 6px;letter-spacing:1px;">\1</div>', fmt)
            fmt = _re.sub(r'##\s+(.*?)(\n|$)', r'<div style="font-family:Orbitron,monospace;font-size:14px;color:var(--gold);margin:14px 0 6px;letter-spacing:1px;">\1</div>', fmt)
            fmt = _re.sub(r'#\s+(.*?)(\n|$)', r'<div style="font-family:Orbitron,monospace;font-size:15px;color:var(--gold);margin:14px 0 8px;letter-spacing:1px;">\1</div>', fmt)
            # Bullet points * item or - item
            fmt = _re.sub(r'(?m)^[\*\-]\s+(.+)$', r'<div style="display:flex;gap:8px;margin:4px 0;"><span style="color:var(--gold);margin-top:2px;">◆</span><span>\1</span></div>', fmt)
            # Newlines
            fmt = fmt.replace('\n\n', '<div style="height:10px;"></div>')
            fmt = fmt.replace('\n', '<br>')

            st.markdown(f"""
            <div class="msg-row-bot">
                <div class="bot-avatar">🤖</div>
                <div class="bubble-bot">
                    <div class="bot-label">UniAssist · NCU</div>
                    {fmt}
                </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # CITATIONS
    if st.session_state.citations:
        with st.expander("📚 Source References", expanded=False):
            for i, chunk in enumerate(st.session_state.citations):
                st.markdown(
                    f'<div class="citation-card"><strong style="color:var(--gold);">📄 Source {i+1}</strong><br><br>{chunk[:380]}...</div>',
                    unsafe_allow_html=True)

    # QUICK SUGGEST CHIPS BELOW CHAT
    st.markdown('<div class="suggest-section-title" style="margin-top:16px;">⚡ Quick Questions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    quick_list = list(quick_map.items())[:6]
    for i, (label, q) in enumerate(quick_list):
        with cols[i % 3]:
            if st.button(label, key=f"chat_sug_{i}", use_container_width=True):
                st.session_state.pending_voice = q
                st.rerun()