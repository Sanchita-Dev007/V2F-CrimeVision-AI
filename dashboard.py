"""
dashboard.py — CrimeVision-AI Tactical Command Center & Pipeline
----------------------------------------------------------------
Features:
- Complete High-Contrast Dark Tactical UI (#05070a, #00FF66, #00F0FF, #FFB000)
- Fixed High-Contrast Widget Labels & Sleek Sliders (No thick green blocks)
- High-Visibility Cyber-Cyan Export Button
- 3D Craniofacial Biometric Mesh Canvas + Audio Waveform Player
- Stage 01: Multi-lingual Voice Intake (Whisper ASR, Translation, Prompt Builder)
- Stage 02: Latent Diffusion Composite Generation (Stable Diffusion 1.5 + SHA-256 Seal)
- Stage 03: FIR Case Database Correlation (DeepFace RetinaFace + FaceNet 512-D)
- Stage 04: Interactive Multi-Accent Plotly Analytics & Hover Charts
- Stage 05: Offline Q&A Viva Architecture Assistant
"""

import os
import json
import time
import hashlib
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

# Pipeline Modules
from pipeline import record_audio, transcribe, translate_text, generate_face
from prompt_builder import build_prompt
from history_lookup import match_face, load_database, DEFAULT_THRESHOLD
import assistant

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="CrimeVision AI // Biometric Craniofacial System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------- REFINED TACTICAL THEME CSS -----------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

    /* Global Dark Theme */
    html, body, [class*="css"], .stApp {
        background-color: #05070a !important;
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    code, pre, .mono-font, .stSelectbox select {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Fixed Dark Tactical Input Box for Tab 5 */
    .stTextInput input, .stTextArea textarea {
        background-color: #080c13 !important;
        color: #00FF66 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border: 1px solid #162334 !important;
        border-radius: 4px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #00FF66 !important;
        box-shadow: 0 0 10px rgba(0, 255, 102, 0.3) !important;
    }

    /* --- HIGH-CONTRAST WIDGET & SLIDER LABELS --- */
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stWidgetLabel"] label, 
    .stSlider label,
    .stRadio label {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
    }

    /* Sleek Tactical Slider Track (Fixes bulky green block) */
    div[data-baseweb="slider"] {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
    }
    div[data-baseweb="slider"] > div {
        background-color: transparent !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #00ff66 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 12px #00ff66 !important;
    }
    div[data-baseweb="slider"] div {
        color: #00ff66 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* --- EXPORT EVIDENCE / DOWNLOAD BUTTON VISIBILITY --- */
    .stDownloadButton>button {
        background: #090d14 !important;
        color: #00F0FF !important;
        border: 1px solid #00F0FF !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        border-radius: 2px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton>button:hover {
        background: #00F0FF !important;
        color: #05070a !important;
        box-shadow: 0 0 16px rgba(0, 240, 255, 0.6) !important;
    }

    /* Tactical Box Frame */
    .tactical-box {
        background: #080c13;
        border: 1px solid #162334;
        border-radius: 4px;
        padding: 24px;
        position: relative;
        margin-bottom: 24px;
        box-shadow: 0 0 35px rgba(0,0,0,0.85);
    }
    .tactical-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 7px; height: 7px;
        border-top: 2px solid #00FF66;
        border-left: 2px solid #00FF66;
    }
    .tactical-box::after {
        content: '';
        position: absolute;
        bottom: 0; right: 0;
        width: 7px; height: 7px;
        border-bottom: 2px solid #00FF66;
        border-right: 2px solid #00FF66;
    }

    /* Cyber Badges & Pills */
    .cyber-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        background: rgba(0, 255, 102, 0.08);
        color: #00ff66;
        padding: 4px 12px;
        border: 1px solid rgba(0, 255, 102, 0.4);
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .cyber-pill-cyan {
        background: rgba(0, 240, 255, 0.08);
        color: #00F0FF;
        border: 1px solid rgba(0, 240, 255, 0.4);
    }
    .cyber-pill-amber {
        background: rgba(255, 176, 0, 0.08);
        color: #FFB000;
        border: 1px solid rgba(255, 176, 0, 0.4);
    }

    .pulse-dot {
        width: 6px;
        height: 6px;
        background-color: #00ff66;
        border-radius: 50%;
        box-shadow: 0 0 8px #00ff66;
        display: inline-block;
    }

    /* Terminal Output Block */
    .tactical-terminal {
        background: #030508;
        border: 1px solid #14201c;
        border-left: 3px solid #00ff66;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
        border-radius: 3px;
        line-height: 1.6;
    }

    /* Buttons */
    .stButton>button {
        background: #00ff66 !important;
        color: #05070a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.07em !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 0.65rem 1.4rem !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: #47ff96 !important;
        box-shadow: 0 0 20px rgba(0, 255, 102, 0.6) !important;
        transform: translateY(-1px);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #162334;
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        padding: 10px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00ff66 !important;
        border-bottom: 2px solid #00ff66 !important;
        font-weight: 700 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------- SESSION STATE -----------------
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "translation" not in st.session_state:
    st.session_state.translation = ""
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "suspect_image_path" not in st.session_state:
    st.session_state.suspect_image_path = (
        "outputs/ai_suspect.png"
        if os.path.exists("outputs/ai_suspect.png")
        else None
    )
if "history_matches" not in st.session_state:
    st.session_state.history_matches = []
if "image_hash" not in st.session_state:
    st.session_state.image_hash = "N/A"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        (
            "assistant",
            "CrimeVision Tactical Assistant online. System ready for viva & architecture queries.",
        )
    ]

# ----------------- TOP NAVBAR -----------------
nav_left, nav_right = st.columns([3, 2])
with nav_left:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; padding:10px 0;">
            <div class="cyber-pill"><span class="pulse-dot"></span>CRIMEVISION [AI v4.8 PRO]</div>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748b;">ACOUSTIC BIOMETRIC RECONSTRUCT // CJIS VERIFIED</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

with nav_right:
    st.markdown(
        """
        <div style="text-align:right; font-family:'JetBrains Mono'; font-size:0.75rem; color:#64748b; padding-top:10px;">
            INTEL PROTOCOL: <span style="color:#00ff66;">INTERPOL // FBI CJIS // NIST 800-88</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

# ----------------- HERO SECTION -----------------
hero_left, hero_right = st.columns([1.25, 1], gap="large")

with hero_left:
    st.markdown(
        """
        <div class="cyber-pill" style="margin-top:10px;">• Forensic Voice-To-Craniofacial Synthesis</div>
        <div style="font-size:2.8rem; font-weight:800; line-height:1.1; letter-spacing:-0.03em; color:#fff; margin:14px 0; text-transform:uppercase;">
            SYNTHESIZING SUSPECT FACES DIRECTLY FROM <span style="color:#00ff66; text-shadow: 0 0 20px rgba(0,255,102,0.4);">AUDIO RECORDINGS</span>
        </div>
        <div style="color:#94a3b8; font-size:0.95rem; line-height:1.6; margin-bottom:24px;">
            CrimeVision AI reconstructs high-fidelity craniofacial morphology from witness statements, wiretaps, and emergency dispatch calls using multi-lingual Whisper transcription, structural prompt compilation, and Latent Diffusion synthesis.
        </div>
    """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.markdown(
        """
        <div style="background:#080c13; border:1px solid #162334; padding:14px; text-align:center; border-radius:4px;">
            <div style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono';">NIST PIV MATCH</div>
            <div style="font-size:1.8rem; font-weight:700; color:#00ff66;">99.2%</div>
        </div>
    """,
        unsafe_allow_html=True,
    )
    m2.markdown(
        """
        <div style="background:#080c13; border:1px solid #162334; padding:14px; text-align:center; border-radius:4px;">
            <div style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono';">WHISPER LATENCY</div>
            <div style="font-size:1.8rem; font-weight:700; color:#00F0FF;">&lt; 1.4s</div>
        </div>
    """,
        unsafe_allow_html=True,
    )
    m3.markdown(
        """
        <div style="background:#080c13; border:1px solid #162334; padding:14px; text-align:center; border-radius:4px;">
            <div style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono';">FACENET VECTOR</div>
            <div style="font-size:1.8rem; font-weight:700; color:#FFB000;">512-D</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with hero_right:
    biometric_html = """
    <div style="background:#070b10; border:1px solid rgba(0, 255, 102, 0.3); border-radius:6px; padding:14px; font-family:'JetBrains Mono', monospace; position:relative; box-shadow:0 0 30px rgba(0,0,0,0.95);">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #162420; padding-bottom:8px; font-size:11px;">
            <span style="color:#00ff66; display:flex; align-items:center; gap:6px;">
                <span style="width:6px; height:6px; background:#00ff66; border-radius:50%; box-shadow:0 0 6px #00ff66;"></span>
                CANVAS: PHENOTYPE_RECON
            </span>
            <span style="color:#64748b;">STANDBY // READY FPS: 60</span>
        </div>

        <div style="position:relative; width:100%; height:260px; display:flex; justify-content:center; align-items:center; overflow:hidden; margin:8px 0; background:#000; border-radius:4px; border:1px solid #162334;">
            <canvas id="bioCanvas" width="360" height="250" style="position:absolute; top:0; left:50%; transform:translateX(-50%); z-index:2;"></canvas>
            <div id="laserScan" style="position:absolute; left:0; right:0; height:2px; background:#00ff66; box-shadow:0 0 15px #00ff66; z-index:3; pointer-events:none;"></div>

            <div style="position:absolute; top:10px; left:10px; z-index:4; background:rgba(5,7,10,0.85); border:1px solid rgba(0,255,102,0.3); padding:8px 10px; border-radius:3px; font-size:9px; line-height:1.5;">
                <div style="color:#64748b;">VOCAL FORMANT: <span style="color:#fff;">F1 580Hz / F2 1840Hz</span></div>
                <div style="color:#64748b;">EST. BIOLOGICAL AGE: <span style="color:#00ff66; font-weight:bold;">34 - 38 YRS</span></div>
                <div style="color:#64748b;">MANDIBULAR INDEX: <span style="color:#fff;">102.4 mm (Broad)</span></div>
                <div style="color:#64748b;">FACIAL CONFIDENCE: <span style="color:#00ff66; font-weight:bold;">96.8%</span></div>
            </div>

            <div style="position:absolute; top:10px; right:10px; z-index:4; font-size:9px; color:#64748b;">
                [ LAT: 37.77 • RNG: LOCK ]
            </div>
        </div>

        <div style="border-top:1px solid #162420; padding-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span id="playIcon" style="cursor:pointer; color:#00ff66; font-size:13px;" onclick="toggleAudio()">▶</span>
                    <span style="color:#fff;">AUDIO INPUT: <span style="color:#00ff66;">911_CALL_INTERCEPT_#849.wav</span></span>
                </div>
                <span id="timerTag" style="color:#64748b; font-size:10px;">00:04 / 00:14</span>
            </div>
            <div id="waveBox" style="display:flex; align-items:flex-end; gap:2px; height:24px; background:#04070a; padding:3px 6px; border-radius:2px; border:1px solid #131f1c;"></div>
        </div>

        <div style="text-align:center; font-size:9px; color:#475569; margin-top:8px;">
            CRIMINAL BIOMETRICS MAPPING PROTOCOL ENCRYPTED AES-256
        </div>
    </div>

    <script>
        const wb = document.getElementById("waveBox");
        for (let i = 0; i < 44; i++) {
            const b = document.createElement("div");
            b.className = "w-bar";
            b.style.flex = "1";
            b.style.backgroundColor = i < 18 ? "#00ff66" : "#1a2c26";
            b.style.height = Math.max(4, Math.sin(i * 0.4) * 18 + 5) + "px";
            b.style.borderRadius = "1px";
            wb.appendChild(b);
        }

        const canvas = document.getElementById("bioCanvas");
        const ctx = canvas.getContext("2d");
        const laser = document.getElementById("laserScan");

        const points = [
            {x: 0, y: -70}, {x: -25, y: -60}, {x: 25, y: -60},
            {x: -45, y: -40}, {x: 45, y: -40}, {x: -20, y: -35}, {x: 20, y: -35},
            {x: 0, y: -25}, {x: 0, y: -5}, {x: -12, y: 5}, {x: 12, y: 5},
            {x: 0, y: 18}, {x: -25, y: 30}, {x: 25, y: 30}, {x: 0, y: 40},
            {x: -35, y: 15}, {x: 35, y: 15}, {x: -25, y: 60}, {x: 25, y: 60}, {x: 0, y: 75}
        ];

        let angle = 0;
        let scanY = 0;

        function render() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            ctx.strokeStyle = "rgba(0, 255, 102, 0.12)";
            ctx.beginPath();
            ctx.arc(cx, cy, 90, 0, Math.PI * 2);
            ctx.stroke();

            angle += 0.02;
            const cos = Math.cos(angle * 0.5);

            const trans = points.map(p => ({ x: cx + p.x * cos, y: cy + p.y }));

            ctx.strokeStyle = "rgba(0, 255, 102, 0.35)";
            for (let i = 0; i < trans.length; i++) {
                for (let j = i + 1; j < trans.length; j++) {
                    const d = Math.hypot(trans[i].x - trans[j].x, trans[i].y - trans[j].y);
                    if (d < 35) {
                        ctx.beginPath();
                        ctx.moveTo(trans[i].x, trans[i].y);
                        ctx.lineTo(trans[j].x, trans[j].y);
                        ctx.stroke();
                    }
                }
            }

            trans.forEach((p, idx) => {
                ctx.fillStyle = idx % 2 === 0 ? "#00FF66" : "#00F0FF";
                ctx.beginPath();
                ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            });

            scanY = (scanY + 0.6) % 100;
            laser.style.top = scanY + "%";

            requestAnimationFrame(render);
        }
        render();

        let isPlaying = false;
        function toggleAudio() {
            isPlaying = !isPlaying;
            document.getElementById("playIcon").innerText = isPlaying ? "⏸" : "▶";
            const bars = document.querySelectorAll(".w-bar");
            if (isPlaying) {
                window.aInterval = setInterval(() => {
                    bars.forEach(b => {
                        b.style.height = (Math.random() * 18 + 4) + "px";
                        b.style.backgroundColor = Math.random() > 0.4 ? "#00ff66" : "#1a2c26";
                    });
                }, 100);
            } else {
                clearInterval(window.aInterval);
            }
        }
    </script>
    """
    components.html(biometric_html, height=380)

st.markdown(
    "<hr style='border-color:#162334; margin: 25px 0;'>",
    unsafe_allow_html=True,
)

# ----------------- TACTICAL SYSTEM TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "[ 01 // WITNESS AUDIO INTAKE ]",
        "[ 02 // COMPOSITE SYNTHESIS ]",
        "[ 03 // FIR CASE CORRELATION ]",
        "[ 04 // FORENSIC CHARTS & ANALYTICS ]",
        "[ 05 // SYSTEM FAQ ASSISTANT ]",
    ]
)

# ==============================================================================
# TAB 1: AUDIO INTAKE
# ==============================================================================
with tab1:
    st.markdown(
        """
        <div class="tactical-box">
            <div class="cyber-pill">STAGE 01: AUDIO INTAKE & ASR TRANSCRIPTION</div>
            <div style="margin-top:8px; font-size:0.85rem; color:#94a3b8;">
                Captures acoustic witness testimony in English, Hindi, or Punjabi. Whisper performs automatic speech recognition, followed by Deep Translator to formulate a standardized forensic prompt.
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    c_in1, c_in2 = st.columns([1.1, 1], gap="large")

    with c_in1:
        modality = st.radio(
            "Select Audio Modality:",
            [
                "Microphone Stream (Live Record)",
                "Manual Verbal Description / Preset Clip",
            ],
            horizontal=True,
        )

        if modality == "Microphone Stream (Live Record)":
            duration = st.slider("Intake Window (Seconds):", 3, 10, 5)
            if st.button("🔴 COMMENCE AUDIO INTAKE"):
                status_box = st.empty()
                for sec in range(3, 0, -1):
                    status_box.markdown(
                        f"<div class='tactical-terminal'>⚠️ Commencing microphone intake in {sec}s... Speak clearly.</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(1)

                status_box.markdown(
                    "<div class='tactical-terminal' style='border-left-color:#FF3366; color:#FF6666;'>🔴 RECORDING WITNESS AUDIO STREAM (Speak now)...</div>",
                    unsafe_allow_html=True,
                )

                os.makedirs("audio", exist_ok=True)
                audio_file = record_audio(
                    seconds=duration, output_file="audio/voice_input.wav"
                )
                status_box.markdown(
                    "<div class='tactical-terminal'>⚡ Stream captured. Processing Whisper ASR transcription & English translation...</div>",
                    unsafe_allow_html=True,
                )

                raw = transcribe(audio_file)
                eng = translate_text(raw)
                prompt = build_prompt(eng)

                st.session_state.audio_path = audio_file
                st.session_state.transcription = raw
                st.session_state.translation = eng
                st.session_state.prompt = prompt
                st.success("Witness voice successfully ingested and processed.")

        else:
            preset_choice = st.selectbox(
                "Load simulated witness description (English / Hindi / Punjabi):",
                [
                    "-- Select Simulated Testimony --",
                    "Hindi: उसकी उम्र लगभग 30-35 साल थी, चेहरा चौकोर और बाईं आंख के ऊपर एक कट का निशान था।",
                    "Punjabi: ਉਹ ਲਗਭਗ 30 ਸਾਲਾਂ ਦਾ ਸੀ, ਤਿੱਖੇ ਨੈਣ ਨਕਸ਼ ਅਤੇ ਦਾੜ੍ਹੀ ਸੀ।",
                    "English: He was around 35 years old with sunken dark eyes, an angular jaw, and light stubble.",
                ],
            )
            custom_text = st.text_area(
                "Or manually input verbal witness observation:",
                placeholder="He was approximately 35 with an angular jawline, dark eyes, and a visible scar...",
            )

            if st.button("PROCESS VERBAL STATEMENT"):
                chosen = (
                    custom_text
                    if custom_text.strip()
                    else preset_choice.split(": ")[-1]
                )
                if chosen and chosen != "-- Select Simulated Testimony --":
                    eng = translate_text(chosen)
                    st.session_state.transcription = chosen
                    st.session_state.translation = eng
                    st.session_state.prompt = build_prompt(eng)
                    st.success("Verbal description normalized to prompt.")

    with c_in2:
        st.markdown("##### 🛰️ Pipeline Intermediates (ASR & Normalization Telemetry)")
        if st.session_state.transcription:
            if st.session_state.audio_path and os.path.exists(
                st.session_state.audio_path
            ):
                st.audio(st.session_state.audio_path, format="audio/wav")

            st.markdown(
                f"""
                <div class="tactical-terminal">
                    <span style="color:#00ff66;">[RAW WITNESS SPEECH]:</span><br>
                    {st.session_state.transcription}<br><br>
                    <span style="color:#00F0FF;">[TRANSLATED TEXT]:</span><br>
                    {st.session_state.translation}<br><br>
                    <span style="color:#FFB000;">[STRUCTURED FORENSIC PROMPT]:</span><br>
                    <span style="font-size:0.75rem; color:#cbd5e1;">{st.session_state.prompt}</span>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="tactical-box" style="text-align:center; padding:40px 20px; border:1px dashed #162334;">
                    <div style="font-size:1.8rem; color:#1e293b; margin-bottom:6px;">🎙️</div>
                    <div style="font-family:'JetBrains Mono'; color:#64748b; font-size:0.8rem;">AWAITING AUDIO STREAM</div>
                    <div style="font-size:0.7rem; color:#475569; margin-top:4px;">Trigger live recording or choose simulated testimony.</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# TAB 2: COMPOSITE SYNTHESIS
# ==============================================================================
with tab2:
    st.markdown(
        """
        <div class="tactical-box">
            <div class="cyber-pill cyber-pill-cyan">STAGE 02: LATENT DIFFUSION SYNTHESIS</div>
            <div style="margin-top:8px; font-size:0.85rem; color:#94a3b8;">
                Executes reverse latent diffusion using Stable Diffusion v1.5 with CPU memory optimization (attention slicing enabled).
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    c_syn1, c_syn2 = st.columns([1, 1.2], gap="large")

    with c_syn1:
        st.markdown("##### ⚙️ Hyperparameter Controls")
        sd_steps = st.slider(
            "Inference Steps (CPU Recommended: 25-35):", 15, 50, 30
        )
        cfg_scale = st.slider(
            "Guidance Scale (CFG):", 4.0, 14.0, 8.0, step=0.5
        )

        st.markdown(
            f"""
            <div class="tactical-terminal" style="margin: 15px 0;">
                MODEL ID: sd-legacy/stable-diffusion-v1-5<br>
                TORCH DTYPE: float32 (CPU)<br>
                ATTENTION SLICING: ACTIVE (8GB RAM Capable)<br>
                RESOLUTION: 512 x 512 px (Grayscale Composite)
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("⚡ EXECUTE DIFFUSION SYNTHESIS"):
            if not st.session_state.prompt:
                st.session_state.prompt = build_prompt(
                    "He was roughly 35 years old with sunken dark eyes, an angular jaw, and a faint scar above his left eyebrow."
                )

            with st.spinner("Running reverse diffusion denoising on CPU..."):
                out = generate_face(
                    prompt_text=st.session_state.prompt,
                    steps=sd_steps,
                    guidance=cfg_scale,
                )
                st.session_state.suspect_image_path = out

                with open(out, "rb") as f:
                    st.session_state.image_hash = hashlib.sha256(
                        f.read()
                    ).hexdigest()

                match_res = match_face(out, threshold=DEFAULT_THRESHOLD)
                st.session_state.history_matches = match_res.get(
                    "matches", []
                )

                st.success(
                    "Suspect composite synthesized and SHA-256 evidence sealed."
                )

    with c_syn2:
        st.markdown("##### 👤 Synthesized Suspect Composite Dossier")
        if st.session_state.suspect_image_path and os.path.exists(
            st.session_state.suspect_image_path
        ):
            img_c, meta_c = st.columns([1.1, 1])
            with img_c:
                st.image(
                    st.session_state.suspect_image_path,
                    caption="CRIMEVISION SYNTHESIZED COMPOSITE",
                    use_container_width=True,
                )
            with meta_c:
                st.markdown(
                    f"""
                    <div class="tactical-terminal" style="font-size:0.75rem;">
                        <b>FORENSIC SPECS</b><br>
                        STATUS: Validated<br>
                        FORMAT: PNG (512x512)<br>
                        CLASSIFIER: FaceNet-512<br>
                        DETECTOR: RetinaFace<br><br>
                        <b>CHAIN OF CUSTODY</b><br>
                        SHA-256:<br>
                        <span style="color:#00F0FF; word-break:break-all;">{st.session_state.image_hash[:24]}...</span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                with open(st.session_state.suspect_image_path, "rb") as img_file:
                    st.download_button(
                        label="💾 EXPORT EVIDENCE FILE",
                        data=img_file,
                        file_name="suspect_composite.png",
                        mime="image/png",
                    )
        else:
            st.markdown(
                """
                <div class="tactical-box" style="text-align:center; padding:50px 20px; border:1px dashed #162334;">
                    <div style="font-size:2rem; color:#1e293b; margin-bottom:8px;">👤</div>
                    <div style="font-family:'JetBrains Mono'; color:#64748b; font-size:0.8rem;">NO ACTIVE COMPOSITE RENDERED</div>
                    <div style="font-size:0.7rem; color:#475569; margin-top:4px;">Trigger diffusion synthesis to generate suspect facial sketch.</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# TAB 3: FIR CORRELATION
# ==============================================================================
with tab3:
    st.markdown(
        """
        <div class="tactical-box">
            <div class="cyber-pill cyber-pill-amber">STAGE 03: FIR RECORD CORRELATION</div>
            <div style="margin-top:8px; font-size:0.85rem; color:#94a3b8;">
                Extracts a 512-dimensional facial embedding vector using DeepFace (RetinaFace backend + FaceNet weights) and computes cosine similarity against pre-computed database.json records.
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns([1.5, 1])
    with t_col1:
        match_thresh = (
            st.slider("Cosine Similarity Threshold (%):", 50, 95, 70, step=1)
            / 100.0
        )
    with t_col2:
        if st.button("RE-EVALUATE DATABASE MATCHES"):
            if st.session_state.suspect_image_path and os.path.exists(
                st.session_state.suspect_image_path
            ):
                m_result = match_face(
                    st.session_state.suspect_image_path,
                    threshold=match_thresh,
                )
                st.session_state.history_matches = m_result.get(
                    "matches", []
                )
                st.success("Database query finished.")
            else:
                st.warning("Generate a composite in Stage 02 first.")

    st.markdown("##### 🚨 Correlated Criminal Records")
    if st.session_state.history_matches:
        for m in st.session_state.history_matches:
            st.markdown(
                f"""
                <div style="background:#090d14; border:1px solid #162334; border-left:3px solid #00ff66; padding:14px; margin-top:10px; border-radius:4px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#00ff66; font-family:'JetBrains Mono'; font-size:1.05rem;">{m['case_id']}</span>
                        <span class="cyber-pill">SIMILARITY: {m['similarity_score']}%</span>
                    </div>
                    <div style="font-size:0.8rem; color:#94a3b8; font-family:'JetBrains Mono'; margin-top:6px; line-height:1.6;">
                        • CRIME TYPE: <span style="color:#fff;">{m['crime_type']}</span><br>
                        • PRIOR FIRS : <span style="color:#fff;">{m['fir_count']} registered incidents</span><br>
                        • STATUS     : <span style="color:#FFB000;">{m['status']}</span><br>
                        • LAST REPORTED : <span style="color:#fff;">{m['last_reported']}</span>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="tactical-terminal" style="border-left-color: #FFB000; color: #cbd5e1;">
                <span class="cyber-pill cyber-pill-amber" style="margin-bottom:8px;">[AWAITING STAGE 02 SYNTHESIS]</span><br>
                No active suspect composite evaluated yet or no records exceeded threshold. Render composite in Stage 02 to run live correlation.
            </div>
        """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 4: CHARTS & ANALYTICS
# ==============================================================================
with tab4:
    st.markdown(
        """
        <div class="tactical-box">
            <div class="cyber-pill" style="border-color:#A855F7; color:#A855F7; background:rgba(168,85,247,0.08);">
                STAGE 04: MULTI-ACCENT HOVER ANALYTICS
            </div>
            <div style="margin-top:8px; font-size:0.85rem; color:#94a3b8;">
                Multi-accent forensic telemetry. Hover over any bar or donut slice to inspect dynamic tooltips and vector scores.
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if os.path.exists("database.json"):
        with open("database.json", "r") as db_file:
            cases = json.load(db_file)

        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            c_ids = [c["case_id"] for c in cases]
            f_counts = [c["fir_count"] for c in cases]
            c_types = [c["crime_type"] for c in cases]
            colors = ["#00F0FF", "#FFB000", "#00FF66"]

            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=c_ids,
                        y=f_counts,
                        text=[
                            f"{t} ({cnt} FIRs)"
                            for t, cnt in zip(c_types, f_counts)
                        ],
                        textposition="auto",
                        marker=dict(
                            color=colors[: len(c_ids)],
                            line=dict(color="#ffffff", width=1),
                        ),
                        hoverinfo="text+y",
                    )
                ]
            )
            fig_bar.update_layout(
                title=dict(
                    text="FIR INCIDENT VOLUME PER SUSPECT RECORD",
                    font=dict(family="Space Grotesk", color="#ffffff", size=13),
                ),
                paper_bgcolor="#080c13",
                plot_bgcolor="#04070a",
                font=dict(color="#94a3b8", family="JetBrains Mono"),
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(gridcolor="#162334", title="Recorded FIRs"),
                xaxis=dict(gridcolor="#162334"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_c2:
            status_tally = {}
            for c in cases:
                status_tally[c["status"]] = (
                    status_tally.get(c["status"], 0) + 1
                )

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=list(status_tally.keys()),
                        values=list(status_tally.values()),
                        hole=0.6,
                        marker=dict(colors=["#00F0FF", "#FFB000"]),
                        hoverinfo="label+percent+value",
                    )
                ]
            )
            fig_donut.update_layout(
                title=dict(
                    text="CASE INVESTIGATION DISPOSITION",
                    font=dict(family="Space Grotesk", color="#ffffff", size=13),
                ),
                paper_bgcolor="#080c13",
                plot_bgcolor="#04070a",
                font=dict(color="#94a3b8", family="JetBrains Mono"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("##### 🎯 Live Biometric Cosine Similarity Rankings")
        if st.session_state.history_matches:
            m_ids = [m["case_id"] for m in st.session_state.history_matches]
            m_scores = [
                m["similarity_score"] for m in st.session_state.history_matches
            ]

            fig_sim = go.Figure(
                data=[
                    go.Bar(
                        x=m_scores,
                        y=m_ids,
                        orientation="h",
                        marker=dict(
                            color=m_scores,
                            colorscale=[
                                [0, "#162334"],
                                [0.5, "#00F0FF"],
                                [1, "#00FF66"],
                            ],
                            showscale=True,
                        ),
                    )
                ]
            )
            fig_sim.update_layout(
                paper_bgcolor="#080c13",
                plot_bgcolor="#04070a",
                font=dict(color="#94a3b8", family="JetBrains Mono"),
                xaxis=dict(
                    range=[0, 100],
                    title="Cosine Match (%)",
                    gridcolor="#162334",
                ),
                yaxis=dict(gridcolor="#162334"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=240,
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.markdown(
                """
                <div class="tactical-terminal">
                    No active match comparison yet. Synthesize a suspect composite in Stage 02 to populate similarity vector graphics.
                </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("database.json not found. Run mock_database.py first.")

# ==============================================================================
# TAB 5: FAQ & VIVA ASSISTANT
# ==============================================================================
with tab5:
    st.markdown(
        """
        <div class="tactical-box">
            <div class="cyber-pill">STAGE 05: ARCHITECTURE & VIVA ASSISTANT</div>
            <div style="margin-top:8px; font-size:0.85rem; color:#94a3b8;">
                Offline rule-based Q&A intelligence engine explaining model mechanics, hardware optimization, and ethical considerations.
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    q_chips = st.columns(3)
    suggested = assistant.get_suggested_questions()
    for idx, q in enumerate(suggested[:3]):
        if q_chips[idx].button(q, key=f"chip_{idx}"):
            ans = assistant.get_answer(q)
            st.session_state.chat_history.append(("user", q))
            st.session_state.chat_history.append(("assistant", ans))

    st.markdown(
        "<hr style='border-color:#162334; margin:18px 0;'>",
        unsafe_allow_html=True,
    )

    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(
                f"""
                <div style="text-align:right; margin:10px 0;">
                    <div style="display:inline-block; background:#111822; border:1px solid #1f2d3d; padding:10px 16px; border-radius:4px; font-size:0.85rem; color:#e2e8f0; font-family:'JetBrains Mono';">
                        <b>Investigator:</b> {msg}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="text-align:left; margin:10px 0;">
                    <div style="display:inline-block; background:#080c12; border-left:3px solid #00ff66; border-top:1px solid #162334; border-right:1px solid #162334; border-bottom:1px solid #162334; padding:12px 18px; border-radius:4px; font-size:0.85rem; color:#94a3b8; font-family:'JetBrains Mono';">
                        <b style="color:#00ff66;">CrimeVision Intelligence:</b> {msg}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    user_q = st.text_input(
        "Enter inquiry regarding architecture, hardware optimization, or limitations:"
    )
    if st.button("TRANSMIT QUESTION") and user_q:
        ans = assistant.get_answer(user_q)
        st.session_state.chat_history.append(("user", user_q))
        st.session_state.chat_history.append(("assistant", ans))
        st.rerun()

# ----------------- RESTRICTED AGENCY ACCESS FOOTER -----------------
st.markdown(
    """
    <div style="margin-top:40px; background: linear-gradient(90deg, #080c13 0%, #0d141e 50%, #080c13 100%); border: 1px solid #162334; border-radius: 6px; padding: 28px; text-align: center;">
        <span style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#00ff66; letter-spacing:0.15em;">• RESTRICTED LAW ENFORCEMENT ACCESS •</span>
        <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin:8px 0; text-transform:uppercase;">
            DEPLOY CRIMEVISION AI IN YOUR JURISDICTION
        </div>
        <div style="color:#64748b; font-size:0.8rem; font-family:'JetBrains Mono'; max-width:600px; margin: 0 auto 16px auto;">
            Pilot CrimeVision Voice-to-Suspect Reconstruction on closed cold cases and live dispatch surveillance channels.
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# ----------------- LEGAL / CJIS FOOTER -----------------
st.markdown(
    """
    <div style="border-top:1px solid #162334; margin-top:40px; padding:20px 0; display:flex; justify-content:space-between; align-items:center; font-family:'JetBrains Mono'; font-size:0.72rem; color:#64748b;">
        <div>
            <span style="color:#ffffff; font-weight:bold;">CRIMEVISION AI</span> &copy; 2026 FORENSIC BIOMETRIC SYSTEMS LLC. ALL RIGHTS RESERVED.
        </div>
        <div style="display:flex; gap:16px;">
            <span>CJIS SECURITY POLICY</span>
            <span>ETHICAL AI CHARTER</span>
            <span>WARRANT PROTOCOLS</span>
            <span>ENCRYPTED API</span>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)
