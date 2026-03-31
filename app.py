import streamlit as st
import os
import pickle
import numpy as np
import json
import base64
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
import google.generativeai as genai

api_key = "AIzaSyDvu0dozL4pOqfS9QO7alR70QkVqX5kceY"
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
logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" '
    f'style="height:52px;margin-right:14px;vertical-align:middle;'
    f'filter:drop-shadow(0 0 8px #C5A57288);" />'
) if logo_b64 else '<span style="font-size:36px;margin-right:12px;">GRD</span>'

# ====================================
# CHECK VOICE QUERY FROM URL PARAMS
# ====================================
params = st.query_params
voice_query_raw = params.get("vq", "")
if voice_query_raw:
    st.query_params.clear()
    if "pending_voice" not in st.session_state:
        st.session_state.pending_voice = voice_query_raw

# ====================================
# STYLES + VOICE PANEL (injected into main page, NOT iframe)
# ====================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {{
    --gold: #C5A572;
    --gold-dim: #7a6544;
    --text-primary: #e8eaf0;
    --text-secondary: #8a9ab8;
    --border: rgba(197,165,114,0.18);
}}

html, body, .stApp {{ background-color: #050a14 !important; color: #e8eaf0 !important; font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
.block-container {{ padding-top: 1.5rem !important; max-width: 1100px; }}

.ncu-header {{
    display:flex; align-items:center; padding:18px 24px;
    background: linear-gradient(135deg,rgba(13,26,46,0.95),rgba(10,18,37,0.95));
    border:1px solid var(--border); border-radius:16px; margin-bottom:24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(197,165,114,0.1);
    position:relative; overflow:hidden;
}}
.ncu-header::after {{
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg,transparent,var(--gold),transparent);
}}
.header-text h1 {{
    font-family:'Cinzel',serif; font-size:26px; font-weight:700;
    color:var(--gold); letter-spacing:1.5px;
    text-shadow: 0 0 20px rgba(197,165,114,0.3);
}}
.header-text p {{ font-size:12.5px; color:var(--text-secondary); margin-top:4px; }}

.user-bubble {{ display:flex; justify-content:flex-end; margin-bottom:14px; animation:sR 0.3s ease; }}
.user-bubble .bubble {{
    background:linear-gradient(135deg,#1a2d4a,#0f2040);
    border:1px solid rgba(30,144,255,0.25); border-radius:18px 18px 4px 18px;
    padding:12px 18px; max-width:72%; color:#cdd8ee; font-size:14.5px; line-height:1.6;
}}
.bot-bubble {{ display:flex; justify-content:flex-start; margin-bottom:16px; animation:sL 0.3s ease; }}
.bot-bubble .avatar {{
    width:36px; height:36px; border-radius:50%;
    background:linear-gradient(135deg,var(--gold),var(--gold-dim));
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0; margin-right:10px; margin-top:2px;
    box-shadow:0 0 12px rgba(197,165,114,0.3);
}}
.bot-bubble .bubble {{
    background:linear-gradient(135deg,#0d1a2e,#091422); border:1px solid var(--border);
    border-radius:4px 18px 18px 18px; padding:14px 18px; max-width:75%;
    color:var(--text-primary); font-size:14.5px; line-height:1.7;
}}
@keyframes sR {{ from{{opacity:0;transform:translateX(16px);}} to{{opacity:1;transform:none;}} }}
@keyframes sL {{ from{{opacity:0;transform:translateX(-16px);}} to{{opacity:1;transform:none;}} }}

/* ---- FLOATING VOICE PANEL ---- */
#va-float {{
    position:fixed; bottom:90px; right:24px; z-index:9999;
    background:linear-gradient(135deg,#0a1628,#060e1c);
    border:1px solid var(--border); border-radius:20px;
    padding:20px 18px; width:230px;
    box-shadow:0 8px 40px rgba(0,0,0,0.8);
    display:none; text-align:center; font-family:'Inter',sans-serif;
    backdrop-filter: blur(10px);
}}
#va-float.open {{ display:block; animation:fadeUp 0.25s ease; }}
@keyframes fadeUp {{ from{{opacity:0;transform:translateY(12px);}} to{{opacity:1;transform:none;}} }}

#va-toggle-btn {{
    position:fixed; bottom:24px; right:24px; z-index:10000;
    width:58px; height:58px; border-radius:50%;
    background:linear-gradient(135deg,#1a2d4a,#0a1220);
    border:2px solid var(--gold-dim); font-size:24px; cursor:pointer;
    box-shadow:0 4px 20px rgba(197,165,114,0.3);
    display:flex; align-items:center; justify-content:center;
    transition:all 0.3s; color:white;
}}
#va-toggle-btn:hover {{ transform:scale(1.08); border-color:var(--gold); box-shadow:0 4px 28px rgba(197,165,114,0.5); }}

#va-orb {{
    width:64px; height:64px; border-radius:50%;
    background:radial-gradient(circle at 35% 35%,#1a2d4a,#060e1c);
    border:2px solid #7a6544; display:flex; align-items:center; justify-content:center;
    font-size:24px; cursor:pointer; margin:0 auto 10px; transition:all 0.3s;
    box-shadow:0 0 18px rgba(197,165,114,0.15);
}}
#va-orb:hover {{ transform:scale(1.06); }}

.sw {{ display:flex; align-items:center; justify-content:center; gap:3px; height:26px; margin:6px 0; }}
.wb {{ width:3px; border-radius:3px; background:var(--gold); opacity:0.2; transition:opacity 0.3s; }}
.wb.on {{ opacity:1; animation:wv 0.7s ease-in-out infinite alternate; }}
.wb:nth-child(1){{height:6px;animation-delay:0s;}}
.wb:nth-child(2){{height:14px;animation-delay:.1s;}}
.wb:nth-child(3){{height:22px;animation-delay:.2s;}}
.wb:nth-child(4){{height:16px;animation-delay:.15s;}}
.wb:nth-child(5){{height:10px;animation-delay:.05s;}}
.wb:nth-child(6){{height:18px;animation-delay:.25s;}}
.wb:nth-child(7){{height:8px;animation-delay:.1s;}}
@keyframes wv {{ from{{transform:scaleY(0.3);}} to{{transform:scaleY(1);}} }}

#va-status {{ font-size:11px; color:#8a9ab8; margin:5px 0 8px 0; min-height:16px; }}
#va-transcript {{
    background:#060e1c; border:1px solid rgba(197,165,114,0.1); border-radius:8px;
    padding:7px 10px; font-size:12px; color:#8a9ab8; min-height:32px;
    margin-bottom:10px; font-style:italic; text-align:left; word-break:break-word;
}}
.vrow {{ display:flex; gap:7px; justify-content:center; flex-wrap:wrap; }}
.vbtn {{
    background:#0d1a2e; border:1px solid rgba(197,165,114,0.18);
    border-radius:7px; padding:6px 11px; color:#C5A572;
    font-size:11.5px; cursor:pointer; transition:all 0.2s; font-family:'Inter',sans-serif;
}}
.vbtn:hover {{ background:#152238; border-color:#C5A572; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background:#060e1c !important; border-right:1px solid var(--border) !important; }}
section[data-testid="stSidebar"] * {{ color:var(--text-primary) !important; }}
.sb-title {{
    font-family:'Cinzel',serif; font-size:11px; color:var(--gold-dim);
    letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;
    padding-bottom:5px; border-bottom:1px solid rgba(197,165,114,0.1);
}}
.stButton > button {{
    background:linear-gradient(135deg,#0d1a2e,#091422) !important;
    color:var(--gold) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; font-family:'Inter',sans-serif !important;
    font-size:13px !important; transition:all 0.2s !important; width:100% !important;
}}
.stButton > button:hover {{
    background:linear-gradient(135deg,#152238,#0d1a2e) !important;
    border-color:var(--gold) !important; transform:translateY(-1px) !important;
}}
.citation-card {{
    background:#0a1628; border:1px solid rgba(197,165,114,0.12);
    border-left:3px solid var(--gold-dim); border-radius:8px;
    padding:12px 16px; margin-bottom:10px; font-size:13px;
    color:var(--text-secondary); line-height:1.6;
}}
* {{ scrollbar-width:thin; scrollbar-color:#1a2d4a transparent; }}
</style>

<!-- FLOATING VOICE BUTTON -->
<button id="va-toggle-btn" onclick="toggleVA()">🎙️</button>

<!-- VOICE PANEL -->
<div id="va-float">
    <div style="font-family:'Cinzel',serif;font-size:11px;color:#C5A572;letter-spacing:2px;margin-bottom:12px;">VOICE ASSISTANT</div>
    <div id="va-orb" onclick="startListen()">🎙️</div>
    <div class="sw">
        <div class="wb" id="w1"></div><div class="wb" id="w2"></div>
        <div class="wb" id="w3"></div><div class="wb" id="w4"></div>
        <div class="wb" id="w5"></div><div class="wb" id="w6"></div>
        <div class="wb" id="w7"></div>
    </div>
    <div id="va-status">Click orb or Speak button</div>
    <div id="va-transcript">Your speech appears here...</div>
    <div class="vrow">
        <button class="vbtn" onclick="startListen()">🎤 Speak</button>
        <button class="vbtn" onclick="stopAll()">⏹ Stop</button>
        <button class="vbtn" id="autobtn" onclick="toggleAuto(this)">🔕 Auto</button>
    </div>
</div>

<script>
var rec = null;
var synth = window.speechSynthesis;
var listening = false;
var autoSpeak = false;
var vaOpen = false;
var lastBotCount = 0;

function toggleVA() {{
    vaOpen = !vaOpen;
    document.getElementById('va-float').classList.toggle('open', vaOpen);
    var btn = document.getElementById('va-toggle-btn');
    btn.textContent = vaOpen ? '✕' : '🎙️';
    btn.style.borderColor = vaOpen ? '#C5A572' : '#7a6544';
}}

function setupRec() {{
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {{
        setS('❌ Use Chrome or Edge for voice');
        return null;
    }}
    var r = new SR();
    r.continuous = false;
    r.interimResults = true;
    r.lang = 'en-IN';
    r.onstart = function() {{
        listening = true;
        setOrb('red');
        setS('🔴 Listening... speak now');
        waves(true);
    }};
    r.onresult = function(e) {{
        var fi = '', intr = '';
        for (var i = e.resultIndex; i < e.results.length; i++) {{
            var t = e.results[i][0].transcript;
            if (e.results[i].isFinal) fi += t; else intr += t;
        }}
        var tr = document.getElementById('va-transcript');
        if (tr) tr.textContent = fi || intr;
        if (fi) window._vaTrans = fi;
    }};
    r.onend = function() {{
        listening = false;
        waves(false);
        if (window._vaTrans) {{
            setS('Sending to UniAssist...');
            sendQuery(window._vaTrans);
            window._vaTrans = null;
        }} else {{
            setOrb('idle');
            setS('Click orb or Speak button');
        }}
    }};
    r.onerror = function(e) {{
        listening = false; waves(false); setOrb('idle');
        var msgs = {{
            'not-allowed': '🚫 Allow microphone access in browser',
            'no-speech': '🔇 No speech detected — try again',
            'network': '🌐 Network error',
            'audio-capture': '🎙️ No microphone found'
        }};
        setS(msgs[e.error] || '⚠️ Error: ' + e.error);
    }};
    return r;
}}

function sendQuery(text) {{
    // Reload page with vq param — Streamlit picks it up via st.query_params
    var url = new URL(window.location.href);
    url.searchParams.set('vq', text);
    window.location.href = url.toString();
}}

function speakText(text) {{
    if (!synth) return;
    synth.cancel();
    setOrb('gold'); setS('🔊 Speaking...'); waves(true);
    var clean = text
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .replace(/\*(.*?)\*/g, '$1')
        .replace(/#{1,6} /g, '')
        .replace(/\n+/g, '. ')
        .replace(/•|-/g, '')
        .replace(/\s+/g, ' ').trim();
    var chunks = [], sents = clean.match(/[^.!?]+[.!?]*/g) || [clean], ch = '';
    sents.forEach(function(s) {{
        if ((ch+s).length < 180) ch += s;
        else {{ if (ch) chunks.push(ch.trim()); ch = s; }}
    }});
    if (ch) chunks.push(ch.trim());
    var idx = 0;
    function next() {{
        if (idx >= chunks.length) {{
            waves(false); setOrb('idle');
            setS('Click orb or Speak button'); return;
        }}
        var u = new SpeechSynthesisUtterance(chunks[idx]);
        u.lang = 'en-IN'; u.rate = 0.93; u.pitch = 1.0; u.volume = 1.0;
        var vv = synth.getVoices();
        var pv = vv.find(function(v) {{ return v.name.includes('Google') && v.lang.startsWith('en'); }})
               || vv.find(function(v) {{ return v.lang.startsWith('en-IN'); }})
               || vv.find(function(v) {{ return v.lang.startsWith('en'); }});
        if (pv) u.voice = pv;
        u.onend = function() {{ idx++; next(); }};
        u.onerror = function() {{ idx++; next(); }};
        synth.speak(u);
    }}
    next();
}}

function stopAll() {{
    if (rec && listening) {{ try {{ rec.stop(); }} catch(e) {{}} }}
    synth.cancel(); waves(false); setOrb('idle');
    setS('Click orb or Speak button');
}}

function toggleAuto(btn) {{
    autoSpeak = !autoSpeak;
    btn.textContent = autoSpeak ? '🔊 Auto' : '🔕 Auto';
    btn.style.color = autoSpeak ? '#C5A572' : '#8a9ab8';
    setS(autoSpeak ? '🔊 Auto-speak ON — will read answers' : '🔕 Auto-speak OFF');
}}

// Watch for new bot bubbles when auto-speak is on
var obs = new MutationObserver(function() {{
    if (!autoSpeak) return;
    var bots = document.querySelectorAll('.bot-bubble .bubble');
    if (bots.length > lastBotCount) {{
        lastBotCount = bots.length;
        var last = bots[bots.length - 1];
        if (last) speakText(last.innerText);
    }}
}});
obs.observe(document.body, {{ childList: true, subtree: true }});

function startListen() {{
    if (listening) return;
    if (!rec) rec = setupRec();
    if (!rec) return;
    window._vaTrans = null;
    var tr = document.getElementById('va-transcript');
    if (tr) tr.textContent = '';
    try {{ rec.start(); }}
    catch(e) {{ rec = setupRec(); if (rec) rec.start(); }}
}}

function setS(m) {{ var el = document.getElementById('va-status'); if (el) el.textContent = m; }}
function waves(on) {{
    for (var i = 1; i <= 7; i++) {{
        var b = document.getElementById('w'+i);
        if (b) {{ if (on) b.classList.add('on'); else b.classList.remove('on'); }}
    }}
}}
function setOrb(s) {{
    var o = document.getElementById('va-orb');
    if (!o) return;
    o.style.border = s==='red'?'2px solid #ff4d6d':s==='gold'?'2px solid #C5A572':'2px solid #7a6544';
    o.style.boxShadow = s==='red'?'0 0 24px rgba(255,77,109,0.5)':s==='gold'?'0 0 24px rgba(197,165,114,0.5)':'0 0 18px rgba(197,165,114,0.15)';
    o.textContent = s==='red'?'🔴':s==='gold'?'🔊':'🎙️';
}}

// Preload voices
if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = function() {{ synth.getVoices(); }};
synth.getVoices();
window._vaTrans = null;
</script>
""", unsafe_allow_html=True)

# ====================================
# HEADER
# ====================================
st.markdown(f"""
<div class="ncu-header">
    {logo_html}
    <div class="header-text">
        <h1>UniAssist NCU</h1>
        <p>The Official AI Assistant of The NorthCap University &middot; Gurugram, Haryana</p>
    </div>
</div>
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
    except Exception as e:
        return [], np.array([]), None, False

chunks, embeddings, embed_model, db_loaded = load_resources()
if not db_loaded:
    st.warning("Vector database not found. Run build_vectorstore.py first.")

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
        logs.append({"query": query})
        with open("analytics.json", "w") as f:
            json.dump(logs, f)
    except:
        pass

# ====================================
# SESSION STATE
# ====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "citations" not in st.session_state:
    st.session_state.citations = []

# ====================================
# SIDEBAR
# ====================================
with st.sidebar:
    if logo_b64:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:center;
                    padding:16px 0 20px;border-bottom:1px solid rgba(197,165,114,0.15);margin-bottom:20px;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="height:60px;filter:drop-shadow(0 0 10px rgba(197,165,114,0.4));" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;padding:20px;font-size:48px;">🎓</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-title">Voice Assistant</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1628;border:1px solid rgba(197,165,114,0.12);border-radius:10px;padding:12px 14px;font-size:12.5px;color:#8a9ab8;line-height:1.7;">
        🎙️ Click the <strong style="color:#C5A572;">mic button</strong> (bottom-right corner) to open voice panel.<br><br>
        🔴 <strong>Speak</strong> your question<br>
        🤖 UniAssist <strong>answers</strong><br>
        🔊 Toggle <strong>Auto</strong> to hear it read back<br><br>
        <span style="color:#4a6080;font-size:11px;">Requires Chrome or Edge browser</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-title" style="margin-top:20px;">Quick Questions</div>', unsafe_allow_html=True)
    quick_map = {
        "🎓 Scholarships": "What scholarships are available at NCU?",
        "💰 Fee Structure": "Explain the fee structure at NCU.",
        "🏢 Placements": "Tell me about placements at NCU.",
        "📚 Courses": "What courses does NCU offer?",
        "🏨 Hostel": "Tell me about hostel facilities at NCU.",
        "📅 Admissions": "How do I apply for admission to NCU?",
    }
    for label, q in quick_map.items():
        if st.button(label, key=f"q_{label}"):
            st.session_state.pending_voice = q

    st.markdown("---")
    st.markdown('<div class="sb-title">Admin</div>', unsafe_allow_html=True)
    if st.button("📊 Analytics"):
        if os.path.exists("analytics.json"):
            with open("analytics.json") as f:
                logs = json.load(f)
            st.metric("Total Queries", len(logs))
        else:
            st.info("No queries yet.")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.citations = []
        st.rerun()

# ====================================
# PROCESS QUERY
# ====================================
query = None

if "pending_voice" in st.session_state:
    query = st.session_state.pending_voice
    del st.session_state.pending_voice

user_input = st.chat_input("Ask anything about NCU...")
if user_input:
    query = user_input

if query:
    log_query(query)
    st.session_state.chat_history.append(("user", query))
    with st.spinner("UniAssist is thinking..."):
        answer, citations = generate_answer(query)
    st.session_state.chat_history.append(("bot", answer))
    st.session_state.citations = citations
    st.rerun()

# ====================================
# CHAT DISPLAY
# ====================================
if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align:center;padding:70px 20px;">
        <div style="font-size:56px;margin-bottom:16px;opacity:0.25;">🎓</div>
        <div style="font-family:'Cinzel',serif;font-size:15px;letter-spacing:1px;
                    margin-bottom:8px;color:#3a5070;">Welcome to UniAssist NCU</div>
        <div style="font-size:13px;color:#2a3d52;">
            Ask anything about admissions, courses, fees, placements, and more.<br>
            <span style="color:#3a4d62;">Use the 🎙️ button (bottom-right) for voice input.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="user-bubble"><div class="bubble">{message}</div></div>',
                        unsafe_allow_html=True)
        else:
            fmt = message.replace('\n', '<br>')
            st.markdown(f'<div class="bot-bubble"><div class="avatar">🤖</div><div class="bubble">{fmt}</div></div>',
                        unsafe_allow_html=True)

    if st.session_state.citations:
        st.markdown("---")
        st.markdown("### 📚 Source References")
        for i, chunk in enumerate(st.session_state.citations):
            st.markdown(
                f'<div class="citation-card"><strong>Source {i+1}</strong><br>{chunk[:350]}...</div>',
                unsafe_allow_html=True
            )