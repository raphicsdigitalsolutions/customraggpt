import hashlib
import html

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from rag_core import build_vectorstore, conversation_history, display_page, format_docs


# ============================================================
# NEXORA Knowledge Studio — stable Streamlit multi-file RAG app
# ============================================================

st.set_page_config(
    page_title="NEXORA | Knowledge Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The theme is scoped to each visitor's Streamlit session.
if "day_mode" not in st.session_state:
    st.session_state.day_mode = False

STAR_ICON = """<svg viewBox="0 0 40 40" aria-hidden="true" focusable="false"><path d="M20 1.5c2.3 10.4 8.1 16.2 18.5 18.5C28.1 22.3 22.3 28.1 20 38.5 17.7 28.1 11.9 22.3 1.5 20 11.9 17.7 17.7 11.9 20 1.5Z" fill="currentColor"/></svg>"""
BRAND_MARK = f'<span class="nx-mark" aria-hidden="true">{STAR_ICON}</span>'

# CSS is intentionally conservative so it doesn't hide Streamlit
# controls or cause unreadable text in different Streamlit themes.
st.markdown(
    """
    <style>
    :root {
      --nx-bg: #090b12;
      --nx-panel: #111522;
      --nx-panel-2: #171c2d;
      --nx-line: rgba(190, 199, 255, .13);
      --nx-text: #f4f6ff;
      --nx-muted: #a8b0c7;
      --nx-violet: #9b8cff;
      --nx-cyan: #65e8f2;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
      background: var(--nx-bg) !important;
      color: var(--nx-text) !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
      background: rgba(9,11,18,.96) !important;
      color: var(--nx-text) !important;
    }
    [data-testid="stHeader"] { border-bottom: 1px solid rgba(255,255,255,.04); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #101421 0%, #0b0e17 100%) !important;
      border-right: 1px solid var(--nx-line);
    }
    [data-testid="stSidebar"] * { color: #e8ebf7; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
      color: #a8b0c7 !important;
    }
    [data-testid="stMain"] .block-container {
      max-width: 1480px;
      padding-top: 2.2rem;
      padding-bottom: 3rem;
    }
    h1, h2, h3, h4, p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
      color: var(--nx-text);
    }
    [data-testid="stCaptionContainer"] { color: var(--nx-muted) !important; }
    [data-testid="stTextInput"] label, [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label, [data-testid="stFileUploader"] label {
      color: #c5cbe0 !important;
      font-weight: 600;
    }
    input, textarea, [data-baseweb="select"] > div {
      background: #171c2a !important;
      color: #f5f6ff !important;
      border-color: rgba(190,199,255,.18) !important;
      border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section {
      background: linear-gradient(145deg, rgba(155,140,255,.09), rgba(101,232,242,.035)) !important;
      border: 1px dashed rgba(155,140,255,.45) !important;
      border-radius: 16px !important;
    }
    [data-testid="stFileUploader"] section * { color: #e8ebf7 !important; }
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
      background: linear-gradient(115deg, #9b8cff, #65dce9) !important;
      color: #0a0d15 !important;
      border: 0 !important;
      border-radius: 12px !important;
      font-weight: 750 !important;
      min-height: 2.7rem;
      box-shadow: 0 7px 22px rgba(110, 104, 245, .18);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 30px rgba(110, 104, 245, .28);
    }
    [data-testid="stSlider"] [role="slider"] { background: #a99bff !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] div { border-color: #a99bff; }
    [data-testid="stAlert"] { border-radius: 13px; }
    div[data-testid="stChatMessage"] {
      border: 1px solid var(--nx-line);
      border-radius: 18px;
      background: linear-gradient(145deg, rgba(255,255,255,.035), rgba(255,255,255,.012));
      padding: .8rem 1rem;
      box-shadow: 0 10px 30px rgba(0,0,0,.10);
    }
    .nx-brand {
      display:flex; align-items:center; gap:.55rem;
      font-size:1.03rem; font-weight:850; letter-spacing:-.035em;
      padding:.55rem 0 1.1rem; border-bottom:1px solid var(--nx-line); margin-bottom:1rem;
    }
    .nx-mark {
      display:inline-flex; align-items:center; justify-content:center;
      width:36px; height:36px; border-radius:12px;
      color:#10131b !important;
      background:linear-gradient(145deg,#c3baff 0%,#8f80ff 45%,#65e8f2 100%);
      box-shadow:0 7px 22px rgba(133,117,255,.32), inset 0 1px 0 rgba(255,255,255,.7);
      font-size:1.2rem;
    }
    .nx-brand-name { color:#f4f6ff !important; }
    .nx-brand-accent { color:#a99bff !important; }
    .nx-hero {
      position:relative; overflow:hidden;
      padding:2.1rem 2.2rem; border-radius:26px; margin:.2rem 0 1.35rem;
      border:1px solid rgba(190,199,255,.18);
      background:
        radial-gradient(ellipse at 90% 5%, rgba(101,232,242,.12), transparent 33%),
        radial-gradient(ellipse at 70% 100%, rgba(155,140,255,.14), transparent 38%),
        linear-gradient(135deg,#171b2c 0%,#111522 65%,#101923 100%);
      box-shadow:0 24px 70px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.055);
    }
    .nx-eyebrow {
      color:var(--nx-cyan); text-transform:uppercase; letter-spacing:.16em;
      font-size:.72rem; font-weight:850;
    }
    .nx-title {
      font-size:clamp(2rem,3.3vw,3.25rem); line-height:1.07; letter-spacing:-.06em;
      font-weight:850; margin:.7rem 0 .9rem; color:#f4f6ff;
    }
    .nx-gradient-text {
      background:linear-gradient(100deg,#b6aaff,#8e82ff 55%,#70e5ee);
      -webkit-background-clip:text; background-clip:text; color:transparent;
    }
    .nx-muted { color:var(--nx-muted) !important; line-height:1.7; }
    .nx-stat {
      padding:1.15rem 1.25rem; border-radius:18px; min-height:102px;
      border:1px solid var(--nx-line);
      background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.018));
      box-shadow:0 12px 30px rgba(0,0,0,.10), inset 0 1px 0 rgba(255,255,255,.035);
    }
    .nx-stat-label { color:#c2cbe0; font-size:.82rem; }
    .nx-stat-value { font-size:1.65rem; font-weight:800; padding-top:.35rem; letter-spacing:-.04em; }
    .nx-login-wrap {
      max-width: 980px; margin: 4vh auto 0;
      display:grid; grid-template-columns:1.05fr .95fr; gap:1.2rem; align-items:stretch;
    }
    .nx-login-art {
      min-height:500px; position:relative; overflow:hidden; border-radius:30px;
      padding:2.2rem; display:flex; flex-direction:column; justify-content:space-between;
      border:1px solid rgba(190,199,255,.19);
      background:
        radial-gradient(ellipse at 25% 20%,rgba(101,232,242,.18),transparent 34%),
        radial-gradient(ellipse at 78% 78%,rgba(155,140,255,.23),transparent 38%),
        linear-gradient(145deg,#1b2035,#0e1320 68%,#111b27);
      box-shadow:0 28px 85px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
    }
    .nx-orb-stage { position:relative; height:245px; display:flex; align-items:center; justify-content:center; perspective:800px; }
    .nx-orb {
      width:180px; height:180px; border-radius:42px; transform:rotateX(17deg) rotateY(-24deg) rotateZ(-14deg);
      background:
        radial-gradient(circle at 28% 20%,rgba(255,255,255,.95),transparent 9%),
        radial-gradient(circle at 30% 25%,#d6d0ff 0%,#8d7cff 19%,#5145a5 43%,#242b4d 68%,#111725 100%);
      box-shadow:inset -18px -20px 35px rgba(0,0,0,.55), inset 8px 8px 22px rgba(255,255,255,.25),
        0 24px 55px rgba(116,101,255,.35), 0 0 80px rgba(101,232,242,.12);
      position:relative; animation:nx-float 6s ease-in-out infinite;
    }
    .nx-orb:before {
      content:""; position:absolute; inset:20px; border-radius:30px;
      border:1px solid rgba(255,255,255,.35); box-shadow:inset 0 0 25px rgba(255,255,255,.08);
    }
    .nx-orb:after {
      content:"✦"; position:absolute; inset:0; display:grid; place-items:center;
      color:#f7f7ff; font-size:4.2rem; text-shadow:0 0 24px rgba(255,255,255,.65);
    }
    .nx-orbit {
      position:absolute; width:270px; height:100px; border:1px solid rgba(101,232,242,.55);
      border-radius:50%; transform:rotate(-27deg); box-shadow:0 0 20px rgba(101,232,242,.09);
    }
    .nx-orbit.two { width:300px; height:125px; transform:rotate(43deg); border-color:rgba(169,155,255,.48); }
    @keyframes nx-float { 0%,100%{transform:translateY(0) rotateX(17deg) rotateY(-24deg) rotateZ(-14deg)} 50%{transform:translateY(-12px) rotateX(22deg) rotateY(-16deg) rotateZ(-9deg)} }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-radius: 26px !important;
      border: 1px solid rgba(190,199,255,.18) !important;
      background: linear-gradient(145deg,rgba(27,32,50,.98),rgba(15,19,30,.98)) !important;
      box-shadow: 0 28px 85px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.055);
      padding: 1rem !important;
    }
    .nx-login-card {
      padding:2rem; border-radius:30px; border:1px solid rgba(190,199,255,.18);
      background:linear-gradient(145deg,rgba(27,32,50,.98),rgba(15,19,30,.98));
      box-shadow:0 28px 85px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.055);
      align-self:center;
    }
    .nx-login-heading { font-size:2rem; font-weight:850; letter-spacing:-.055em; margin:.5rem 0; color:#f4f6ff; }
    .nx-login-note { color:var(--nx-muted); line-height:1.65; margin-bottom:1.4rem; }
    .nx-pill {
      display:inline-block; padding:.35rem .65rem; border-radius:999px;
      border:1px solid rgba(101,232,242,.25); background:rgba(101,232,242,.07);
      color:#8beaf0; font-size:.72rem; font-weight:750; letter-spacing:.04em;
    }
    /* High-contrast readability patch */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
      color: #f4f6ff !important;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stHelp"],
    [data-testid="stHelp"] *,
    label, small, .stCaption, .stMarkdown p {
      color: #d7dcef !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
      color: #cbd2e6 !important;
      opacity: 1 !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
      background: #1a2030 !important;
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
      border: 1px solid #59627a !important;
      caret-color: #75e8f2 !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
      color: #9ba6bf !important;
      -webkit-text-fill-color: #9ba6bf !important;
      opacity: 1 !important;
    }
    [data-baseweb="select"] > div {
      background: #1a2030 !important;
      border: 1px solid #59627a !important;
    }
    [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [role="listbox"] *,
    [role="option"] {
      color: #f4f6ff !important;
    }
    [data-baseweb="popover"], [role="listbox"] {
      background: #171c2a !important;
      border: 1px solid #59627a !important;
    }
    [role="option"]:hover, [aria-selected="true"] {
      background: #303751 !important;
    }
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] button {
      color: #f4f6ff !important;
      opacity: 1 !important;
    }
    [data-testid="stSlider"] *,
    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
      color: #e0e5f5 !important;
      opacity: 1 !important;
    }
    [data-testid="stAlert"] *,
    [data-testid="stExpander"] *,
    [data-testid="stChatMessage"] *,
    [data-testid="stMetric"] * {
      opacity: 1 !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
      color: #f4f6ff !important;
    }
    .nx-muted, .nx-login-note, .nx-stat-label {
      color: #c2cbe0 !important;
      opacity: 1 !important;
    }
    .nx-eyebrow { color: #75edf5 !important; }
    .nx-pill { color: #8ff5fb !important; }
    .nx-brand-name { color: #f7f8ff !important; }
    .nx-brand-accent { color: #b5aaff !important; }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
      color: #111522 !important;
    }

    /* Animated premium dashboard stage */
    .nx-main-hero {
      display:grid; grid-template-columns:minmax(0,1.25fr) minmax(250px,.75fr);
      gap:1.25rem; align-items:center; min-height:290px;
      padding:2.15rem 2.35rem; isolation:isolate;
    }
    .nx-main-hero:before {
      content:""; position:absolute; inset:-45%; z-index:-1; pointer-events:none;
      background:conic-gradient(from 0deg, transparent 0 35%,rgba(127,113,255,.13) 44%,rgba(92,230,242,.10) 50%,transparent 62% 100%);
      animation:nx-aurora-turn 22s linear infinite;
    }
    .nx-main-hero:after {
      content:""; position:absolute; inset:0; pointer-events:none; opacity:.28; z-index:-1;
      background-image:radial-gradient(rgba(208,216,255,.28) .7px,transparent .7px);
      background-size:23px 23px; mask-image:linear-gradient(90deg,transparent 20%,#000 100%);
    }
    .nx-hero-copy {position:relative; z-index:2;}
    .nx-hero-title {
      color:#f6f7ff; font-weight:850; font-size:clamp(2.2rem,3.5vw,3.45rem);
      letter-spacing:-.065em; line-height:1.02; margin:.9rem 0 1rem;
    }
    .nx-hero-description {max-width:610px; color:#c7cde0; line-height:1.75; font-size:.98rem;}
    .nx-hero-badges {display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1.35rem;}
    .nx-hero-badge {
      font-size:.72rem; font-weight:700; letter-spacing:.025em; color:#dce2f5;
      padding:.43rem .72rem; border-radius:999px; background:rgba(255,255,255,.045);
      border:1px solid rgba(198,207,255,.15); box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
    }
    .nx-live-dot {display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:.45rem;
      background:#65e8c1;box-shadow:0 0 0 4px rgba(101,232,193,.10),0 0 14px rgba(101,232,193,.8);
      animation:nx-live-pulse 2s ease-in-out infinite;vertical-align:middle;}
    .nx-hero-stage {
      position:relative; min-height:235px; display:flex; align-items:center; justify-content:center;
      perspective:1000px; isolation:isolate;
    }
    .nx-hero-aura {
      position:absolute;width:225px;height:225px;border-radius:50%;
      background:radial-gradient(circle,rgba(135,115,255,.31),rgba(70,213,236,.10) 43%,transparent 70%);
      filter:blur(9px); animation:nx-aura-breathe 5.5s ease-in-out infinite;
    }
    .nx-hero-ring {
      position:absolute;width:240px;height:100px;border:1px solid rgba(104,225,242,.63);
      border-radius:50%;transform:rotate(-27deg);box-shadow:0 0 26px rgba(91,210,239,.10),inset 0 0 16px rgba(91,210,239,.06);
      animation:nx-ring-float 7s ease-in-out infinite;
    }
    .nx-hero-ring.ring-2 {width:265px;height:120px;border-color:rgba(173,156,255,.6);transform:rotate(52deg);animation-delay:-2.2s;}
    .nx-hero-ring.ring-3 {width:200px;height:190px;border-color:rgba(255,255,255,.16);transform:rotate(-60deg);animation-delay:-4s;}
    .nx-hero-crystal {
      position:relative;width:126px;height:126px;border-radius:32px;z-index:2;
      transform-style:preserve-3d;transform:rotateX(24deg) rotateY(-27deg) rotateZ(-12deg);
      background:
        radial-gradient(circle at 27% 17%,rgba(255,255,255,.95),transparent 8%),
        linear-gradient(145deg,#d5ceff 0%,#9384ff 22%,#5d56b7 48%,#242c4a 75%,#111726 100%);
      box-shadow:inset 9px 9px 22px rgba(255,255,255,.35),inset -20px -21px 32px rgba(0,0,0,.55),
        0 25px 38px rgba(0,0,0,.36),0 0 55px rgba(131,112,255,.3);
      animation:nx-crystal-hover 6.8s ease-in-out infinite,nx-crystal-turn 18s linear infinite;
    }
    .nx-hero-crystal:before {
      content:"";position:absolute;inset:13px;border-radius:23px;border:1px solid rgba(255,255,255,.38);
      box-shadow:inset 0 0 20px rgba(255,255,255,.12);transform:translateZ(11px);
    }
    .nx-hero-crystal:after {
      content:"✦";position:absolute;inset:0;display:grid;place-items:center;color:#fff;
      font-size:3.25rem;text-shadow:0 0 25px rgba(255,255,255,.75);transform:translateZ(22px);
    }
    .nx-particle {position:absolute;width:6px;height:6px;border-radius:50%;background:#a9fbff;
      box-shadow:0 0 15px #65e8f2;animation:nx-particle-drift 6s ease-in-out infinite;}
    .nx-particle.p-1 {top:18%;left:18%;animation-delay:-1s;}
    .nx-particle.p-2 {top:25%;right:13%;width:4px;height:4px;animation-delay:-3s;background:#b7aaff;}
    .nx-particle.p-3 {bottom:17%;left:27%;width:5px;height:5px;animation-delay:-4s;background:#b7aaff;}
    .nx-particle.p-4 {bottom:19%;right:18%;animation-delay:-2s;}
    .nx-stage-label {
      position:absolute;bottom:-.15rem;left:50%;transform:translateX(-50%);white-space:nowrap;
      color:#bfc8e1;font-size:.64rem;font-weight:800;letter-spacing:.11em;
      padding:.42rem .68rem;border-radius:999px;border:1px solid rgba(187,199,255,.16);
      background:rgba(12,16,26,.70);box-shadow:0 8px 22px rgba(0,0,0,.18);backdrop-filter:blur(12px);
    }
    .nx-stat {transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;position:relative;overflow:hidden;}
    .nx-stat:before {content:"";position:absolute;left:0;right:0;top:0;height:2px;background:linear-gradient(90deg,#a99bff,#65e8f2);opacity:.7;}
    .nx-stat:hover {transform:translateY(-4px);border-color:rgba(155,140,255,.38);box-shadow:0 18px 38px rgba(0,0,0,.19),0 0 30px rgba(116,101,255,.06);}
    .nx-stat-value {color:#f8f8ff;text-shadow:0 0 22px rgba(160,145,255,.12);}
    .nx-empty-state {
      display:flex;gap:1rem;align-items:center;padding:1.25rem 1.4rem;margin:1.2rem 0;
      border:1px solid rgba(155,140,255,.23);border-radius:18px;
      background:linear-gradient(120deg,rgba(155,140,255,.09),rgba(101,232,242,.035));
    }
    .nx-empty-icon {flex:0 0 50px;width:50px;height:50px;border-radius:16px;display:grid;place-items:center;
      background:linear-gradient(140deg,#b5aaff,#65dce9);color:#10131b;font-size:1.5rem;
      box-shadow:0 9px 24px rgba(134,117,255,.22);}
    @keyframes nx-aurora-turn {to{transform:rotate(360deg)}}
    @keyframes nx-live-pulse {0%,100%{opacity:1;box-shadow:0 0 0 4px rgba(101,232,193,.1),0 0 14px rgba(101,232,193,.7)}50%{opacity:.58;box-shadow:0 0 0 7px rgba(101,232,193,.025),0 0 5px rgba(101,232,193,.25)}}
    @keyframes nx-aura-breathe {0%,100%{transform:scale(.92);opacity:.65}50%{transform:scale(1.12);opacity:1}}
    @keyframes nx-ring-float {0%,100%{translate:0 0;opacity:.65}50%{translate:0 -9px;opacity:1}}
    @keyframes nx-crystal-hover {0%,100%{translate:0 0}50%{translate:0 -12px}}
    @keyframes nx-crystal-turn {0%,100%{filter:hue-rotate(0deg)}50%{filter:hue-rotate(9deg)}}
    @keyframes nx-particle-drift {0%,100%{transform:translate3d(0,0,0);opacity:.45}50%{transform:translate3d(9px,-17px,15px);opacity:1}}
    @media (prefers-reduced-motion: reduce) {
      *, *:before, *:after {animation-duration:.01ms !important;animation-iteration-count:1 !important;scroll-behavior:auto !important;transition-duration:.01ms !important;}
    }

    @media (max-width: 800px) {
      .nx-login-wrap { grid-template-columns:1fr; margin-top:1rem; }
      .nx-login-art { min-height:280px; }
      .nx-orb-stage { height:150px; }
      .nx-orb { width:120px; height:120px; border-radius:30px; }
      .nx-orb:after { font-size:3rem; }
      .nx-hero { padding:1.5rem; }
      .nx-main-hero { grid-template-columns:1fr; gap:.2rem; }
      .nx-hero-stage { min-height:205px; }
      .nx-hero-title { font-size:2.35rem; }
      .nx-stage-label { bottom:.1rem; font-size:.58rem; }
      [data-testid="stMain"] .block-container { padding-top:1.2rem; }
    }
    /* NEXORA's deep-forest theme with legible controls and dimensional surfaces */
    :root {
      --nx-bg: #0b100f;
      --nx-panel: #111a17;
      --nx-panel-2: #18231f;
      --nx-line: rgba(205, 224, 207, .14);
      --nx-text: #f3f2e9;
      --nx-muted: #b7c1b9;
      --nx-violet: #cbe58b;
      --nx-cyan: #8dd4ae;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
      background:
        radial-gradient(ellipse at 78% -12%, rgba(104, 145, 112, .11), transparent 38%),
        #0b100f !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
      background: rgba(11, 16, 15, .96) !important;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #121b17 0%, #0d1311 100%) !important;
      border-right-color: rgba(205, 224, 207, .13);
    }
    [data-testid="stMain"] .block-container { max-width: 1480px; }
    [data-testid="stFileUploader"] section {
      background: linear-gradient(145deg, rgba(203, 229, 139, .075), rgba(141, 212, 174, .035)) !important;
      border-color: rgba(203, 229, 139, .43) !important;
    }
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
      background: linear-gradient(112deg, #d5e99e, #a7d6ad) !important;
      color: #152019 !important;
      -webkit-text-fill-color: #152019 !important;
      box-shadow: 0 8px 24px rgba(133, 171, 108, .17), inset 0 1px 0 rgba(255, 255, 255, .6);
    }
    .stButton > button *, .stDownloadButton > button *, [data-testid="stFormSubmitButton"] button * {
      color: #152019 !important;
      -webkit-text-fill-color: #152019 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
      box-shadow: 0 12px 30px rgba(133, 171, 108, .25), inset 0 1px 0 rgba(255, 255, 255, .68);
    }
    .stButton > button:disabled, [data-testid="stFormSubmitButton"] button:disabled {
      background: #344238 !important;
      color: #edf2df !important;
      -webkit-text-fill-color: #edf2df !important;
      opacity: 1 !important;
      cursor: not-allowed;
      box-shadow: none !important;
    }
    .stButton > button:disabled *, [data-testid="stFormSubmitButton"] button:disabled * {
      color: #edf2df !important;
      -webkit-text-fill-color: #edf2df !important;
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
    [data-testid="stFormSubmitButton"] button:focus-visible,
    input:focus-visible, textarea:focus-visible, [role="slider"]:focus-visible {
      outline: 3px solid #d4ed9b !important;
      outline-offset: 3px !important;
    }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
      background: #18231f !important;
      background-color: #18231f !important;
      color: #f3f2e9 !important;
      -webkit-text-fill-color: #f3f2e9 !important;
      border-color: rgba(205, 224, 207, .3) !important;
      caret-color: #cbe58b !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
      color: #aebbb0 !important;
      -webkit-text-fill-color: #aebbb0 !important;
      opacity: 1 !important;
    }
    [data-testid="stFileUploader"] section button {
      background: #557044 !important;
      border: 1px solid #829b6d !important;
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stFileUploader"] section button * {
      color: #ffffff !important;
      -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
      background: #18231f !important;
      background-color: #18231f !important;
      border-color: rgba(205, 224, 207, .28) !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] svg { fill: #f3f2e9 !important; }
    [data-baseweb="popover"], [role="listbox"] {
      background: #17211c !important;
      border: 1px solid rgba(205, 224, 207, .28) !important;
    }
    [data-baseweb="popover"] *, [role="listbox"] *, [role="option"] { color: #f3f2e9 !important; }
    [role="option"]:hover, [aria-selected="true"] { background: #293a30 !important; }
    .nx-hero {
      border-color: rgba(205, 224, 207, .16);
      background:
        radial-gradient(ellipse at 82% 12%, rgba(141, 212, 174, .13), transparent 35%),
        radial-gradient(ellipse at 20% 110%, rgba(203, 229, 139, .11), transparent 45%),
        linear-gradient(135deg, #18221e 0%, #111a17 62%, #121b18 100%);
      box-shadow: 0 28px 75px rgba(0, 0, 0, .27), inset 0 1px 0 rgba(255, 255, 255, .06);
    }
    .nx-main-hero:before {
      background: conic-gradient(from 0deg, transparent 0 35%, rgba(203, 229, 139, .12) 44%, rgba(141, 212, 174, .09) 50%, transparent 62% 100%);
    }
    .nx-main-hero:after { background-image: radial-gradient(rgba(205, 224, 207, .22) .7px, transparent .7px); }
    .nx-hero-title { color: #f3f2e9; }
    .nx-hero-description { color: #c6d0c7; }
    .nx-hero-badge {
      color: #e0e8db;
      border-color: rgba(205, 224, 207, .16);
      background: rgba(225, 237, 218, .045);
    }
    .nx-live-dot {
      background: #cbe58b;
      box-shadow: 0 0 0 4px rgba(203, 229, 139, .1), 0 0 14px rgba(203, 229, 139, .72);
    }
    .nx-hero-aura {
      background: radial-gradient(circle, rgba(199, 224, 144, .27), rgba(106, 161, 126, .1) 43%, transparent 70%);
    }
    .nx-hero-ring {
      border-color: rgba(141, 212, 174, .56);
      box-shadow: 0 0 26px rgba(141, 212, 174, .1), inset 0 0 16px rgba(141, 212, 174, .06);
    }
    .nx-hero-ring.ring-2 { border-color: rgba(203, 229, 139, .55); }
    .nx-hero-ring.ring-3 { border-color: rgba(238, 241, 220, .16); }
    .nx-hero-crystal {
      background:
        radial-gradient(circle at 27% 17%, rgba(255, 255, 255, .95), transparent 8%),
        linear-gradient(145deg, #e5efc7 0%, #bdd995 22%, #668867 48%, #2b4436 75%, #111a15 100%);
      box-shadow: inset 9px 9px 22px rgba(255, 255, 255, .27), inset -20px -21px 32px rgba(0, 0, 0, .55),
        0 25px 38px rgba(0, 0, 0, .36), 0 0 55px rgba(123, 164, 111, .24);
    }
    .nx-particle { background: #d5ef9e; box-shadow: 0 0 15px rgba(203, 229, 139, .85); }
    .nx-particle.p-2, .nx-particle.p-3 { background: #8dd4ae; }
    .nx-stage-label {
      color: #d9e6cc;
      border-color: rgba(223, 233, 211, .17);
      background: rgba(12, 20, 16, .78);
    }
    .nx-stat { border-color: rgba(205, 224, 207, .13); }
    .nx-stat:before { background: linear-gradient(90deg, #cbe58b, #8dd4ae); }
    .nx-stat:hover { border-color: rgba(203, 229, 139, .32); }
    .nx-stat-value { color: #f3f2e9; text-shadow: 0 0 22px rgba(203, 229, 139, .1); }
    .nx-empty-state {
      border-color: rgba(203, 229, 139, .18);
      background: linear-gradient(120deg, rgba(203, 229, 139, .07), rgba(141, 212, 174, .035));
    }
    .nx-empty-icon {
      background: linear-gradient(140deg, #d5e99e, #a7d6ad);
      color: #152019;
      box-shadow: 0 9px 24px rgba(133, 171, 108, .18);
    }
    .nx-login-art {
      border-color: rgba(205, 224, 207, .17);
      background:
        radial-gradient(ellipse at 25% 20%, rgba(141, 212, 174, .16), transparent 34%),
        radial-gradient(ellipse at 78% 78%, rgba(203, 229, 139, .13), transparent 38%),
        linear-gradient(145deg, #1a241e, #0e1512 68%, #111b16);
    }
    .nx-orb {
      background:
        radial-gradient(circle at 28% 20%, rgba(255, 255, 255, .92), transparent 9%),
        radial-gradient(circle at 30% 25%, #e5efc7 0%, #bdd995 19%, #668867 43%, #2b4436 68%, #111725 100%);
      box-shadow: inset -18px -20px 35px rgba(0, 0, 0, .56), inset 8px 8px 22px rgba(255, 255, 255, .22),
        0 24px 55px rgba(123, 164, 111, .25), 0 0 80px rgba(141, 212, 174, .1);
    }
    .nx-orbit { border-color: rgba(141, 212, 174, .52); }
    .nx-orbit.two { border-color: rgba(203, 229, 139, .45); }
    .nx-mark {
      background: linear-gradient(145deg, #e3efb9 0%, #cbe58b 52%, #82c9a4 100%);
      box-shadow: 0 7px 22px rgba(166, 202, 130, .2), inset 0 1px 0 rgba(255, 255, 255, .7);
    }
    .nx-brand-accent, .nx-gradient-text {
      background: linear-gradient(100deg, #f3edc9 0%, #cbe58b 55%, #8dd4ae 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent !important;
    }
    .nx-pill {
      border-color: rgba(203, 229, 139, .3);
      background: rgba(203, 229, 139, .075);
      color: #d7edaa !important;
    }
    .nx-eyebrow { color: #d5e99e !important; }
    .nx-muted, .nx-login-note, .nx-stat-label, .nx-key-note { color: #c2cbc0 !important; }
    .nx-key-note { font-size: .84rem; line-height: 1.65; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        scroll-behavior: auto !important;
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Additional interactions layer above the original visual system. Keeping it
# separate makes the light theme and motion states easy to reason about.
st.markdown(
    """
    <style>
    .nx-mark svg {display:block;width:20px;height:20px;overflow:visible;}
    .nx-mark {color:#fbffe9 !important;}
    .nx-topbar-label {display:flex;align-items:center;gap:.7rem;font-size:.82rem;font-weight:800;letter-spacing:.08em;color:#e7efd9;}
    .nx-topbar-label small {display:block;margin-top:.15rem;color:#9eab9f;font-size:.66rem;font-weight:600;letter-spacing:.05em;}
    .nx-top-orbit {position:relative;width:58px;height:58px;margin-left:auto;perspective:500px;isolation:isolate;}
    .nx-top-orbit:before {content:"";position:absolute;inset:10px;border-radius:50%;background:radial-gradient(circle,rgba(203,229,139,.22),transparent 70%);filter:blur(4px);animation:nx-top-aura 4.8s ease-in-out infinite;}
    .nx-top-ring {position:absolute;inset:12px 2px;border:1px solid rgba(203,229,139,.5);border-radius:50%;transform:rotate(-30deg);animation:nx-top-spin 8s linear infinite;}
    .nx-top-ring.ring-b {inset:4px 16px;border-color:rgba(141,212,174,.55);transform:rotate(54deg);animation-duration:6s;animation-direction:reverse;}
    .nx-top-core {position:absolute;left:17px;top:17px;width:25px;height:25px;display:grid;place-items:center;border:1px solid rgba(239,246,218,.4);border-radius:9px;background:linear-gradient(145deg,#5e805e,#263d31);color:#f8ffe9;box-shadow:inset 1px 1px 4px rgba(255,255,255,.35),0 6px 16px rgba(112,156,105,.28);transform:rotateX(22deg) rotateY(-24deg) rotateZ(-12deg);transform-style:preserve-3d;animation:nx-top-float 4.5s ease-in-out infinite;}
    .nx-top-core svg {width:15px;height:15px;filter:drop-shadow(0 0 4px rgba(255,255,255,.75));}
    .nx-hero-stage {min-height:270px;overflow:visible;}
    .nx-hero-crystal {width:142px;height:142px;border-radius:35px;transform-style:preserve-3d;}
    .nx-hero-crystal:after {content:none !important;}
    .nx-crystal-mark {position:absolute;inset:0;display:grid;place-items:center;transform:translateZ(24px);}
    .nx-crystal-mark:before {content:"";position:absolute;width:70px;height:70px;border-radius:23px;border:1px solid rgba(239,246,218,.4);background:linear-gradient(145deg,rgba(17,31,23,.88),rgba(37,62,47,.74));box-shadow:inset 0 1px 0 rgba(255,255,255,.2),0 8px 20px rgba(9,19,12,.35);}
    .nx-crystal-mark svg {position:relative;width:37px;height:37px;color:#fbffe8;filter:drop-shadow(0 0 9px rgba(245,255,206,.78));animation:nx-star-breathe 3.6s ease-in-out infinite;}
    .nx-float-chip {position:absolute;z-index:3;padding:.48rem .62rem;border:1px solid rgba(231,242,216,.24);border-radius:10px;background:linear-gradient(145deg,rgba(35,54,42,.86),rgba(18,31,24,.76));color:#eff6df;font-size:.6rem;font-weight:850;letter-spacing:.13em;box-shadow:0 8px 20px rgba(7,16,10,.22),inset 0 1px 0 rgba(255,255,255,.12);transform-style:preserve-3d;animation:nx-chip-float 5.2s ease-in-out infinite;}
    .nx-float-chip.chip-a {top:17%;left:5%;transform:rotateY(18deg) rotateZ(-8deg);}
    .nx-float-chip.chip-b {top:26%;right:3%;animation-delay:-2s;transform:rotateY(-18deg) rotateZ(7deg);}
    .nx-float-chip.chip-c {bottom:19%;left:14%;animation-delay:-3.4s;transform:rotateX(16deg) rotateZ(5deg);}
    .nx-file-shelf {margin:1.6rem 0 1.25rem;}
    .nx-file-shelf-head {display:flex;justify-content:space-between;align-items:end;gap:1rem;margin:0 0 .9rem;}
    .nx-file-shelf-title {margin:.25rem 0 0;font-size:1.25rem;font-weight:800;letter-spacing:-.035em;color:#f3f2e9;}
    .nx-file-count {padding:.38rem .62rem;border:1px solid rgba(205,224,207,.16);border-radius:10px;background:rgba(225,237,218,.05);color:#cfdbca;font-size:.72rem;font-weight:750;}
    .nx-file-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem;perspective:1200px;}
    .nx-file-card {min-width:0;perspective:900px;transform-style:preserve-3d;animation:nx-card-enter .65s cubic-bezier(.22,.78,.24,1) both;animation-delay:calc(var(--card-index,0) * 65ms);}
    .nx-file-card-inner {position:relative;min-height:122px;overflow:hidden;padding:1rem;border:1px solid rgba(205,224,207,.14);border-radius:18px;background:linear-gradient(145deg,rgba(33,47,38,.88),rgba(17,27,22,.94));box-shadow:0 10px 26px rgba(0,0,0,.15),inset 0 1px 0 rgba(255,255,255,.055);transform-style:preserve-3d;transition:transform .42s cubic-bezier(.2,.8,.2,1),border-color .3s ease,box-shadow .3s ease;}
    .nx-file-card-inner:before {content:"";position:absolute;inset:-50% auto -50% -70%;width:42%;background:linear-gradient(90deg,transparent,rgba(235,247,214,.12),transparent);transform:skewX(-20deg);animation:nx-file-sheen 8s ease-in-out infinite;animation-delay:calc(var(--card-index,0) * -450ms);pointer-events:none;}
    .nx-file-card:hover .nx-file-card-inner,.nx-file-card:focus-within .nx-file-card-inner {transform:rotateX(5deg) rotateY(-5deg) translateZ(10px);border-color:rgba(203,229,139,.46);box-shadow:0 22px 38px rgba(0,0,0,.22),0 0 24px rgba(155,190,115,.1),inset 0 1px 0 rgba(255,255,255,.11);}
    .nx-file-card-top {display:flex;justify-content:space-between;align-items:center;gap:.6rem;transform:translateZ(14px);}
    .nx-file-glyph {width:33px;height:38px;display:grid;place-items:center;border:1px solid rgba(203,229,139,.22);border-radius:10px;background:linear-gradient(145deg,rgba(203,229,139,.16),rgba(141,212,174,.08));color:#d9eca8;}
    .nx-file-glyph svg {width:19px;height:22px;}
    .nx-file-type {padding:.3rem .45rem;border-radius:7px;background:rgba(203,229,139,.10);color:#d8e9b1;font-size:.61rem;font-weight:850;letter-spacing:.12em;}
    .nx-file-name {position:relative;z-index:1;display:block;margin:.8rem 0 0;overflow-wrap:anywhere;color:#f2f4e9;font-size:.9rem;font-weight:700;line-height:1.4;transform:translateZ(18px);}
    .nx-file-meta {display:block;margin-top:.27rem;color:#aeb9ac;font-size:.68rem;letter-spacing:.045em;text-transform:uppercase;transform:translateZ(10px);}
    @keyframes nx-top-spin {to{transform:rotate(330deg)}}
    @keyframes nx-top-float {0%,100%{translate:0 1px;rotate:0deg}50%{translate:0 -5px;rotate:5deg}}
    @keyframes nx-top-aura {0%,100%{opacity:.42;scale:.86}50%{opacity:.95;scale:1.12}}
    @keyframes nx-chip-float {0%,100%{translate:0 0}50%{translate:0 -8px}}
    @keyframes nx-star-breathe {0%,100%{scale:.92;opacity:.88}50%{scale:1.08;opacity:1}}
    @keyframes nx-card-enter {from{opacity:0;translate:0 14px}to{opacity:1;translate:0 0}}
    @keyframes nx-file-sheen {0%,74%,100%{opacity:0;translate:0 0}80%{opacity:.7}94%{opacity:0;translate:270% 0}}
    @media (max-width: 900px) {.nx-file-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
    @media (max-width: 600px) {
      .nx-topbar-label {font-size:.7rem;}
      .nx-top-orbit {width:48px;height:48px;}
      .nx-top-core {left:14px;top:14px;width:22px;height:22px;}
      .nx-file-grid {grid-template-columns:1fr;}
      .nx-file-shelf-head {align-items:start;}
      .nx-hero-stage {min-height:220px;}
      .nx-hero-crystal {width:118px;height:118px;border-radius:30px;}
      .nx-crystal-mark:before {width:60px;height:60px;border-radius:19px;}
    }
    @media (prefers-reduced-motion: reduce) {
      .nx-top-orbit *, .nx-top-orbit:before, .nx-hero-stage *, .nx-file-card, .nx-file-card-inner:before {
        animation-duration:.01ms !important;animation-iteration-count:1 !important;scroll-behavior:auto !important;transition-duration:.01ms !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.day_mode:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {background:#f3f5ed !important;color:#243229 !important;color-scheme:light;}
        [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {background:rgba(243,245,237,.95) !important;color:#243229 !important;}
        [data-testid="stSidebar"] {background:linear-gradient(160deg,#e8eddf,#f5f6ef) !important;border-right-color:rgba(59,79,61,.16) !important;}
        [data-testid="stSidebar"] *, [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
        [data-testid="stMain"] label, [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] span:not(.nx-gradient-text),
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span:not(.nx-brand-accent):not(.nx-mark) {color:#26352b !important;-webkit-text-fill-color:#26352b !important;}
        [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#536456 !important;}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#536456 !important;-webkit-text-fill-color:#536456 !important;}
        .nx-topbar-label, .nx-brand-name, .nx-login-heading, .nx-title {color:#26352b !important;}
        [data-testid="stSidebar"] .nx-brand-name {color:#26352b !important;-webkit-text-fill-color:#26352b !important;}
        .nx-topbar-label small {color:#627164 !important;}
        .nx-eyebrow {color:#4e7048 !important;}.nx-pill {color:#47623e !important;background:rgba(185,214,136,.2) !important;}
        .nx-file-shelf-title {color:#25342a !important;}.nx-file-count {color:#465747 !important;background:#fbfcf6 !important;border-color:rgba(67,94,69,.18) !important;}
        .nx-hero {border-color:rgba(73,99,72,.19) !important;background:radial-gradient(ellipse at 82% 12%,rgba(141,176,112,.18),transparent 35%),radial-gradient(ellipse at 18% 110%,rgba(203,229,139,.25),transparent 45%),linear-gradient(135deg,#f8faef,#e9efdf 64%,#f2f5e9) !important;box-shadow:0 26px 68px rgba(55,76,52,.13),inset 0 1px 0 rgba(255,255,255,.85) !important;}
        .nx-main-hero:before {background:conic-gradient(from 0deg,transparent 0 35%,rgba(136,169,94,.15) 44%,rgba(108,165,127,.12) 50%,transparent 62% 100%) !important;}
        .nx-main-hero:after {background-image:radial-gradient(rgba(85,111,79,.20) .7px,transparent .7px) !important;}
        .nx-hero-title {color:#24352a !important;}.nx-hero-description {color:#4a5b4c !important;}
        .nx-gradient-text, .nx-brand-accent {background:linear-gradient(100deg,#526e4c,#6f8c56 58%,#598f76) !important;background-clip:text !important;-webkit-background-clip:text !important;color:transparent !important;}
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] .nx-gradient-text,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] .nx-brand-accent,
        [data-testid="stSidebar"] .nx-brand-accent {color:transparent !important;-webkit-text-fill-color:transparent !important;}
        .nx-hero-badge {color:#354a39 !important;background:rgba(255,255,255,.58) !important;border-color:rgba(83,110,79,.18) !important;}
        .nx-hero-ring {border-color:rgba(82,126,89,.6) !important;box-shadow:0 0 24px rgba(93,142,92,.12),inset 0 0 14px rgba(93,142,92,.08) !important;}
        .nx-hero-ring.ring-2 {border-color:rgba(161,184,105,.68) !important;}.nx-hero-ring.ring-3 {border-color:rgba(68,87,64,.23) !important;}
        .nx-hero-aura {background:radial-gradient(circle,rgba(198,219,143,.55),rgba(135,190,151,.18) 44%,transparent 72%) !important;}
        .nx-hero-crystal {background:radial-gradient(circle at 27% 17%,rgba(255,255,255,.98),transparent 8%),linear-gradient(145deg,#e5edbb 0%,#b8d28e 24%,#708f6c 50%,#375844 76%,#1a2a20 100%) !important;box-shadow:inset 9px 9px 22px rgba(255,255,255,.55),inset -18px -18px 30px rgba(16,34,22,.52),0 22px 38px rgba(63,91,59,.24),0 0 54px rgba(122,160,103,.3) !important;}
        .nx-stage-label {color:#405543 !important;background:rgba(250,252,243,.9) !important;border-color:rgba(73,99,72,.2) !important;}
        .nx-stat {background:linear-gradient(145deg,#fffef8,#edf2e5) !important;border-color:rgba(68,93,67,.17) !important;box-shadow:0 12px 28px rgba(55,76,52,.09),inset 0 1px 0 #fff !important;}
        .nx-stat-label {color:#556555 !important;}.nx-stat-value {color:#24352a !important;text-shadow:0 0 18px rgba(135,164,98,.12) !important;}
        .nx-empty-state {background:linear-gradient(120deg,rgba(203,229,139,.2),rgba(141,212,174,.1)) !important;border-color:rgba(86,121,72,.21) !important;}
        .nx-empty-state .nx-empty-title {color:#26352b !important;}.nx-empty-state .nx-muted {color:#536456 !important;}
        .nx-file-card-inner {background:linear-gradient(145deg,#fffef9,#edf2e5) !important;border-color:rgba(70,96,68,.18) !important;box-shadow:0 12px 28px rgba(54,75,52,.10),inset 0 1px 0 #fff !important;}
        .nx-file-name {color:#26352b !important;}.nx-file-meta {color:#617062 !important;}.nx-file-type {color:#425a3f !important;background:rgba(140,170,103,.16) !important;}.nx-file-glyph {color:#526e4c !important;border-color:rgba(88,117,74,.2) !important;background:linear-gradient(145deg,rgba(203,229,139,.30),rgba(141,212,174,.15)) !important;}
        .nx-login-art {background:radial-gradient(ellipse at 25% 20%,rgba(141,212,174,.25),transparent 36%),radial-gradient(ellipse at 78% 78%,rgba(203,229,139,.28),transparent 40%),linear-gradient(145deg,#f2f5e9,#e4ecdf 70%,#f7f7ee) !important;border-color:rgba(73,99,72,.2) !important;}
        .nx-login-heading {color:#26352b !important;}.nx-login-note,.nx-muted,.nx-stat-label {color:#536456 !important;}
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-baseweb="select"] > div {background:#fffef9 !important;background-color:#fffef9 !important;color:#243229 !important;-webkit-text-fill-color:#243229 !important;border-color:rgba(67,94,69,.28) !important;}
        [role="combobox"] {background:#fffef9 !important;background-color:#fffef9 !important;color:#243229 !important;-webkit-text-fill-color:#243229 !important;}
        [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder {color:#788879 !important;-webkit-text-fill-color:#788879 !important;}
        [data-testid="stFileUploader"] section {background:linear-gradient(145deg,rgba(203,229,139,.2),rgba(141,212,174,.1)) !important;border-color:rgba(83,112,74,.5) !important;}
        [data-testid="stFileUploader"] section *, [data-testid="stFileUploader"] section button {color:#26352b !important;-webkit-text-fill-color:#26352b !important;}
        .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {color:#16251b !important;-webkit-text-fill-color:#16251b !important;background:linear-gradient(115deg,#d9ebad,#9bcaa8) !important;}
        [data-baseweb="popover"], [role="listbox"] {background:#fffef9 !important;border-color:rgba(67,94,69,.25) !important;}
        [data-baseweb="popover"] *, [role="listbox"] *, [role="option"] {color:#243229 !important;}
        [role="option"]:hover, [aria-selected="true"] {background:#e7efdc !important;}
        div[data-testid="stChatMessage"] {background:linear-gradient(145deg,#fffefb,#eef2e8) !important;border-color:rgba(67,94,69,.18) !important;}
        [data-testid="stAlert"] *, [data-testid="stExpander"] * {color:#26352b;}
        .nx-file-card-inner:before {background:linear-gradient(90deg,transparent,rgba(105,145,85,.16),transparent);}
        .nx-top-ring {border-color:rgba(97,137,87,.6) !important;}.nx-top-ring.ring-b {border-color:rgba(102,160,128,.65) !important;}
        /* Keep intentionally styled dashboard spans above the broad day-mode text rule. */
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] span.nx-float-chip {
          color:#26352b !important;-webkit-text-fill-color:#26352b !important;
          background:linear-gradient(145deg,#fffef9,#e7efdf) !important;
          border-color:rgba(73,99,72,.28) !important;
          box-shadow:0 8px 20px rgba(55,76,52,.15),inset 0 1px 0 #fff !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] span.nx-file-name {
          color:#24352a !important;-webkit-text-fill-color:#24352a !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] span.nx-file-meta {
          color:#536456 !important;-webkit-text-fill-color:#536456 !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] span.nx-file-type {
          color:#425a3f !important;-webkit-text-fill-color:#425a3f !important;
          background:rgba(140,170,103,.18) !important;
        }
        [data-testid="stTextInputRootElement"] {
          background:#fffef9 !important;border:1px solid rgba(67,94,69,.28) !important;
        }
        [data-testid="stTextInputRootElement"] input {
          background:transparent !important;color:#243229 !important;-webkit-text-fill-color:#243229 !important;
        }
        [data-testid="stTextInputRootElement"] button {
          color:#425a3f !important;background:transparent !important;border-color:transparent !important;box-shadow:none !important;
        }
        [data-testid="stTextInputRootElement"] button *,
        [data-testid="stTextInputRootElement"] [data-testid="stIconMaterial"] {
          color:#425a3f !important;-webkit-text-fill-color:#425a3f !important;fill:currentColor !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Compact, single-screen workspace and the NEXORA research companion.
st.markdown(
    """
    <style>
    [data-testid="stMain"] .block-container {max-width:1600px;padding-top:.55rem;padding-bottom:.45rem;}
    [data-testid="stMain"] [data-testid="stVerticalBlock"] {gap:.55rem;}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.55rem;}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {padding:.7rem !important;}
    .nx-topbar-label {padding:.25rem 0 .55rem;font-size:.76rem;}
    .nx-topbar-label small {font-size:.61rem;}
    .nx-top-orbit {width:48px;height:48px;}
    .nx-top-core {left:14px;top:14px;width:21px;height:21px;}
    .nx-main-hero {grid-template-columns:minmax(0,1.25fr) minmax(220px,.75fr);gap:.6rem;min-height:205px;padding:1.25rem 1.55rem;margin:.15rem 0 .35rem;}
    .nx-hero-title {font-size:clamp(1.9rem,3vw,2.8rem);margin:.48rem 0 .55rem;}
    .nx-hero-description {max-width:570px;font-size:.88rem;line-height:1.55;}
    .nx-hero-badges {gap:.4rem;margin-top:.75rem;}
    .nx-hero-badge {font-size:.64rem;padding:.34rem .55rem;}
    .nx-hero-stage {min-height:178px;}
    .nx-stage-label {bottom:-.1rem;font-size:.55rem;padding:.33rem .55rem;}
    .nx-stat {min-height:66px;padding:.58rem .78rem;border-radius:14px;}
    .nx-stat-label {font-size:.72rem;}
    .nx-stat-value {font-size:1.3rem;padding-top:.12rem;}
    .nx-empty-state {gap:.7rem;padding:.68rem .85rem;margin:.4rem 0;border-radius:15px;}
    .nx-empty-icon {width:40px;height:40px;flex-basis:40px;border-radius:13px;font-size:1.2rem;}
    .nx-file-shelf {margin:.45rem 0 .25rem;}
    .nx-file-shelf-head {align-items:center;margin-bottom:.45rem;}
    .nx-file-shelf-title {font-size:1rem;}
    .nx-file-grid {display:flex;gap:.6rem;overflow-x:auto;overflow-y:hidden;padding:.1rem .1rem .45rem;scrollbar-width:thin;}
    .nx-file-card {flex:0 0 clamp(155px,18vw,220px);}
    .nx-file-card-inner {min-height:78px;padding:.62rem .75rem;border-radius:14px;}
    .nx-file-card-top {gap:.4rem;}
    .nx-file-glyph {width:28px;height:31px;border-radius:8px;}
    .nx-file-glyph svg {width:16px;height:18px;}
    .nx-file-type {font-size:.55rem;padding:.25rem .38rem;}
    .nx-file-name {margin:.42rem 0 0;font-size:.76rem;line-height:1.25;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
    .nx-file-meta {margin-top:.12rem;font-size:.56rem;}
    .st-key-nx-chat-history {min-height:0;max-height:clamp(7rem,calc(100dvh - 580px),20rem);overflow-y:auto;overscroll-behavior:contain;padding-right:.25rem;scrollbar-width:thin;}
    [data-testid="stMain"] div[data-testid="stChatMessage"] {padding:.48rem .7rem;margin:.22rem 0;border-radius:14px;}
    [data-testid="stMain"] [data-testid="stChatInput"] {padding-top:.25rem;}
    .nx-robot {--nx-bot-shell-hi:#e3edcf;--nx-bot-shell:#9db58d;--nx-bot-shell-deep:#3c5645;--nx-bot-face:#102019;--nx-bot-eye:#e2ffad;--nx-bot-glow:rgba(150,197,129,.34);position:relative;width:180px;height:176px;display:grid;place-items:center;perspective:720px;isolation:isolate;}
    .nx-robot-aura {position:absolute;width:155px;height:145px;border-radius:50%;background:radial-gradient(circle,var(--nx-bot-glow),transparent 70%);filter:blur(9px);animation:nx-bot-aura 4.8s ease-in-out infinite;}
    .nx-robot-orbit {position:absolute;width:230px;height:86px;border:1px solid rgba(141,212,174,.58);border-radius:50%;transform:rotate(-24deg);animation:nx-bot-orbit 11s linear infinite;}
    .nx-robot-orbit.orbit-b {width:148px;height:208px;border-color:rgba(203,229,139,.52);transform:rotate(34deg);animation-duration:14s;animation-direction:reverse;}
    .nx-robot-figure {position:relative;width:126px;height:168px;transform-style:preserve-3d;animation:nx-bot-hover 5.3s ease-in-out infinite;}
    .nx-robot-antenna {position:absolute;left:61px;top:0;width:4px;height:23px;border-radius:5px;background:linear-gradient(90deg,var(--nx-bot-shell-deep),var(--nx-bot-shell-hi));box-shadow:0 1px 3px rgba(0,0,0,.35);transform:translateZ(16px);}
    .nx-robot-antenna:after {content:"";position:absolute;left:-5px;top:-7px;width:14px;height:14px;border-radius:50%;background:radial-gradient(circle at 34% 28%,#fff,#d7ff9d 34%,#73a971 76%);box-shadow:0 0 14px var(--nx-bot-glow);animation:nx-bot-beacon 2.4s ease-in-out infinite;}
    .nx-robot-ear {position:absolute;top:48px;width:13px;height:27px;border:1px solid rgba(239,246,218,.44);border-radius:7px;background:linear-gradient(90deg,var(--nx-bot-shell-deep),var(--nx-bot-shell));box-shadow:0 5px 10px rgba(0,0,0,.28);transform:translateZ(10px);}
    .nx-robot-ear.left {left:7px;}.nx-robot-ear.right {right:7px;}
    .nx-robot-head {position:absolute;left:17px;top:20px;width:92px;height:74px;border:1px solid rgba(239,246,218,.48);border-radius:24px;background:radial-gradient(circle at 28% 12%,rgba(255,255,255,.9),transparent 18%),linear-gradient(145deg,var(--nx-bot-shell-hi),var(--nx-bot-shell) 48%,var(--nx-bot-shell-deep));box-shadow:inset 5px 5px 12px rgba(255,255,255,.38),inset -8px -9px 15px rgba(10,22,14,.48),0 14px 24px rgba(0,0,0,.32);transform:translateZ(20px) rotateX(7deg) rotateY(-9deg);}
    .nx-robot-face {position:absolute;inset:12px 11px;border:1px solid rgba(197,239,177,.4);border-radius:15px;background:radial-gradient(ellipse at 50% 0,rgba(74,139,99,.22),transparent 68%),linear-gradient(145deg,#14271e,var(--nx-bot-face));box-shadow:inset 0 3px 11px rgba(0,0,0,.7),0 1px 0 rgba(255,255,255,.08);display:flex;align-items:center;justify-content:center;gap:15px;transform:translateZ(7px);}
    .nx-robot-eye {width:10px;height:14px;border-radius:50%;background:var(--nx-bot-eye);box-shadow:0 0 7px var(--nx-bot-eye),0 0 15px rgba(194,245,145,.75);animation:nx-bot-blink 5.2s ease-in-out infinite;}
    .nx-robot-eye:nth-child(2) {animation-delay:.08s;}
    .nx-robot-mouth {position:absolute;bottom:8px;width:18px;height:3px;border-radius:5px;background:rgba(186,233,151,.9);box-shadow:0 0 7px rgba(186,233,151,.6);}
    .nx-robot-neck {position:absolute;left:51px;top:91px;width:25px;height:12px;border-radius:4px;background:linear-gradient(90deg,#304838,#d6e7c2 48%,#455e49);transform:translateZ(8px);}
    .nx-robot-torso {position:absolute;left:24px;top:99px;width:78px;height:51px;border:1px solid rgba(239,246,218,.4);border-radius:14px 14px 20px 20px;background:radial-gradient(circle at 30% 4%,rgba(255,255,255,.7),transparent 20%),linear-gradient(145deg,var(--nx-bot-shell-hi),var(--nx-bot-shell) 40%,var(--nx-bot-shell-deep));box-shadow:inset 4px 4px 10px rgba(255,255,255,.3),inset -8px -8px 13px rgba(10,22,14,.43),0 11px 18px rgba(0,0,0,.28);display:grid;place-items:center;transform:translateZ(13px) rotateX(-4deg);}
    .nx-robot-chest {width:25px;height:25px;display:grid;place-items:center;border:1px solid rgba(235,248,219,.48);border-radius:9px;background:linear-gradient(145deg,#233e2e,#15271d);color:#e5ffa9;box-shadow:inset 0 1px 3px rgba(255,255,255,.28),0 0 13px var(--nx-bot-glow);}
    .nx-robot-chest svg {width:13px;height:13px;filter:drop-shadow(0 0 5px rgba(225,255,173,.75));}
    .nx-robot-arm {position:absolute;top:106px;width:18px;height:39px;border:1px solid rgba(239,246,218,.35);border-radius:10px;background:linear-gradient(145deg,var(--nx-bot-shell-hi),var(--nx-bot-shell-deep));transform-origin:top center;box-shadow:inset 2px 1px 4px rgba(255,255,255,.23),0 5px 8px rgba(0,0,0,.25);}
    .nx-robot-arm.left {left:12px;transform:rotate(13deg);animation:nx-bot-arm-left 3.8s ease-in-out infinite;}.nx-robot-arm.right {right:12px;transform:rotate(-13deg);animation:nx-bot-arm-right 4.2s ease-in-out infinite;}
    .nx-robot-leg {position:absolute;top:146px;width:20px;height:20px;border-radius:5px 5px 9px 9px;background:linear-gradient(145deg,var(--nx-bot-shell),var(--nx-bot-shell-deep));box-shadow:inset 2px 1px 4px rgba(255,255,255,.18),0 6px 9px rgba(0,0,0,.24);transform:translateZ(7px);}
    .nx-robot-leg.left {left:38px;}.nx-robot-leg.right {right:38px;}
    @keyframes nx-bot-hover {0%,100%{transform:translateY(1px) rotateX(2deg) rotateY(-5deg)}50%{transform:translateY(-7px) rotateX(-2deg) rotateY(5deg)}}
    @keyframes nx-bot-orbit {to{rotate:360deg}}
    @keyframes nx-bot-aura {0%,100%{opacity:.48;scale:.9}50%{opacity:.92;scale:1.1}}
    @keyframes nx-bot-beacon {0%,100%{scale:.88;filter:brightness(.95)}50%{scale:1.12;filter:brightness(1.25)}}
    @keyframes nx-bot-blink {0%,42%,48%,100%{scale:1 1}45%{scale:1 .12}}
    @keyframes nx-bot-arm-left {0%,100%{rotate:-4deg}50%{rotate:8deg}}
    @keyframes nx-bot-arm-right {0%,100%{rotate:4deg}50%{rotate:-8deg}}
    @media (min-width:1024px) and (min-height:680px) {
      html,body,.stApp,[data-testid="stAppViewContainer"] {height:100dvh !important;overflow:hidden !important;}
      [data-testid="stMain"] {height:calc(100dvh - 3.6rem) !important;overflow:hidden !important;}
      [data-testid="stMain"] .block-container {height:100%;max-height:100%;overflow:hidden !important;box-sizing:border-box;}
      [data-testid="stSidebar"],[data-testid="stSidebar"] > div:first-child {height:100dvh;max-height:100dvh;overflow-y:auto;}
    }
    @media (max-width:1180px) {
      .nx-main-hero {grid-template-columns:minmax(0,1fr) 205px;padding:1.1rem 1.2rem;}
      .nx-robot {scale:.9;}
      .nx-hero-title {font-size:clamp(1.8rem,3.2vw,2.4rem);}
    }
    @media (max-width:900px) {
      .nx-main-hero {grid-template-columns:1fr;min-height:0;}
      .nx-hero-stage {min-height:160px;}
      .nx-robot {scale:.82;}
      [data-testid="stMain"] .block-container {padding-top:.55rem;padding-bottom:4.2rem;}
    }
    @media (prefers-reduced-motion: reduce) {
      .nx-robot *, .nx-robot:before, .nx-robot:after {animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------- Session state --------------------------

DEFAULTS = {
    # Require a key per Streamlit session; a server-wide secret must not unlock
    # the public app for every visitor.
    "api_key": "",
    "vectorstore": None,
    "file_sig": None,
    "file_names": [],
    "messages": [],
    "indexed_chunks": 0,
    "indexed_pages": 0,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

topbar_brand, topbar_theme, topbar_art = st.columns([5.6, 1.25, 0.65])
with topbar_brand:
    st.markdown(
        f'<div class="nx-topbar-label">{BRAND_MARK}<span>NEXORA<small>KNOWLEDGE STUDIO</small></span></div>',
        unsafe_allow_html=True,
    )
with topbar_theme:
    st.toggle("Day mode", key="day_mode", help="Switch between the light and night color themes.")
with topbar_art:
    st.markdown(
        f'<div class="nx-top-orbit" aria-hidden="true"><span class="nx-top-ring"></span><span class="nx-top-ring ring-b"></span><span class="nx-top-core">{STAR_ICON}</span></div>',
        unsafe_allow_html=True,
    )


# -------------------------- API key gate ----------------------------
# Use a reliable in-page gate instead of st.dialog so this works on
# older Streamlit versions as well and never leaves a blank page.

if not st.session_state.api_key:
    art_col, login_col = st.columns([1.05, 0.95], gap="large")

    with art_col:
        st.markdown(
            f"""
            <section class="nx-login-art">
              <div>
                <div class="nx-brand" style="border:0;margin:0;padding:0">
                  {BRAND_MARK}
                  <span class="nx-brand-name">NEXORA</span>
                  <span class="nx-brand-accent">STUDIO</span>
                </div>
                <div class="nx-eyebrow" style="margin-top:2.2rem">Private AI workspace</div>
                <h1 class="nx-title" style="font-size:2.5rem;margin-top:.7rem">Your knowledge.<br><span class="nx-gradient-text">One intelligent space.</span></h1>
                <p class="nx-muted">Turn your documents into useful answers, with source passages you can verify.</p>
              </div>
              <div class="nx-orb-stage">
                <div class="nx-orbit"></div><div class="nx-orbit two"></div><div class="nx-orb"></div>
              </div>
              <div class="nx-muted" style="font-size:.78rem">PDF · DOCX · CSV · JSON · TXT · MARKDOWN</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with login_col:
        with st.container(border=True):
            st.markdown(
                """
                <div style="padding:.5rem 0 .8rem">
                  <span class="nx-pill">✦ YOUR OPENAI CONNECTION</span>
                  <h2 class="nx-login-heading" style="margin-top:1.1rem">Welcome back.</h2>
                  <p class="nx-login-note">Use your own OpenAI API key to search your documents. Your key stays in this active app session and is sent to OpenAI for the requests you make.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.form("first_api_key_form", border=False):
                key_input = st.text_input(
                    "OpenAI API key",
                    type="password",
                    placeholder="Paste your API key",
                    help="Used for OpenAI requests in this session. It is never shown after submission.",
                )
                connect = st.form_submit_button("Connect workspace  →", use_container_width=True)
            st.markdown(
                '<p class="nx-key-note">NEXORA does not save your key to chat history or uploaded files.</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="nx-muted" style="font-size:.78rem;margin-top:1.6rem">SUPPORTED FILES<br>PDF · DOCX · CSV · JSON · TXT · MD</p>',
                unsafe_allow_html=True,
            )

    if connect:
        if key_input.strip():
            st.session_state.api_key = key_input.strip()
            st.rerun()
        else:
            st.warning("Please enter your API key.")
    st.stop()

# ------------------------------- Sidebar ----------------------------

with st.sidebar:
    st.markdown(
        f'<div class="nx-brand">{BRAND_MARK}<span class="nx-brand-name">NEXORA</span><span class="nx-brand-accent">STUDIO</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Your multi-file AI research workspace")

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md", "csv", "json", "docx"],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, Markdown, CSV, JSON, DOCX",
    )

    st.divider()
    st.markdown("#### Model & retrieval")
    model_name = st.selectbox(
        "Chat model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )
    with st.expander("Advanced retrieval", expanded=False):
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Retrieved passages", 2, 10, 4)

    st.divider()
    with st.expander("API key settings"):
        replacement_key = st.text_input(
            "Replace API key",
            type="password",
            placeholder="Enter new key",
        )
        if st.button("Update API key", use_container_width=True):
            if replacement_key.strip():
                st.session_state.api_key = replacement_key.strip()
                st.success("API key updated for this session.")
                st.rerun()
            else:
                st.warning("Enter a replacement key first.")

        if st.button("Disconnect API key", use_container_width=True):
            st.session_state.api_key = ""
            st.session_state.vectorstore = None
            st.session_state.file_sig = None
            st.session_state.file_names = []
            st.session_state.messages = []
            st.rerun()

    col1, col2 = st.columns(2)
    if col1.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col2.button("Reset files", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.file_sig = None
        st.session_state.file_names = []
        st.session_state.messages = []
        st.session_state.indexed_chunks = 0
        st.session_state.indexed_pages = 0
        st.rerun()


# -------------------------- Index uploaded files --------------------

# If the user removes all uploads from the widget, clear the old knowledge base too.
if not uploaded_files and st.session_state.file_names:
    st.session_state.vectorstore = None
    st.session_state.file_sig = None
    st.session_state.file_names = []
    st.session_state.messages = []
    st.session_state.indexed_chunks = 0
    st.session_state.indexed_pages = 0

if uploaded_files:
    file_parts = []
    for file in uploaded_files:
        digest = hashlib.sha256(file.getvalue()).hexdigest()
        file_parts.append(f"{file.name}:{digest}")
    files_digest = hashlib.sha256("\n".join(sorted(file_parts)).encode("utf-8")).hexdigest()
    signature = (files_digest, chunk_size, chunk_overlap)

    if st.session_state.file_sig != signature:
        st.session_state.vectorstore = None
        st.session_state.file_sig = None
        st.session_state.messages = []
        try:
            with st.spinner("Reading documents and building your search index..."):
                vectorstore, chunk_count, page_count, file_names = build_vectorstore(
                    uploaded_files,
                    chunk_size,
                    chunk_overlap,
                    st.session_state.api_key,
                    on_warning=st.warning,
                )
            st.session_state.vectorstore = vectorstore
            st.session_state.file_sig = signature
            st.session_state.file_names = file_names
            st.session_state.indexed_chunks = chunk_count
            st.session_state.indexed_pages = page_count
            st.success(f"Indexed {len(file_names)} file(s).")
        except Exception as exc:
            st.error(f"Could not index files: {exc}")
            st.stop()


# ------------------------------ Main UI -----------------------------

st.markdown(
    f"""
    <section class="nx-hero nx-main-hero">
      <div class="nx-hero-copy">
        <div class="nx-eyebrow"><span class="nx-live-dot"></span>Knowledge engine online</div>
        <h1 class="nx-hero-title">Know more.<br><span class="nx-gradient-text">Search smarter.</span></h1>
        <p class="nx-hero-description">Your documents, transformed into an intelligent knowledge space. Ask precise questions, explore source passages, and stay in control of every answer.</p>
        <div class="nx-hero-badges">
          <span class="nx-hero-badge">✦ Multi-file RAG</span>
          <span class="nx-hero-badge">⌁ Source-grounded answers</span>
          <span class="nx-hero-badge">◈ Live retrieval</span>
        </div>
      </div>
      <div class="nx-hero-stage" aria-hidden="true">
        <div class="nx-hero-aura"></div>
        <div class="nx-hero-ring ring-1"></div><div class="nx-hero-ring ring-2"></div><div class="nx-hero-ring ring-3"></div>
        <div class="nx-robot">
          <div class="nx-robot-aura"></div>
          <div class="nx-robot-orbit"></div><div class="nx-robot-orbit orbit-b"></div>
          <div class="nx-robot-figure">
            <div class="nx-robot-antenna"></div>
            <div class="nx-robot-ear left"></div><div class="nx-robot-ear right"></div>
            <div class="nx-robot-head">
              <div class="nx-robot-face"><span class="nx-robot-eye"></span><span class="nx-robot-eye"></span><span class="nx-robot-mouth"></span></div>
            </div>
            <div class="nx-robot-neck"></div>
            <div class="nx-robot-arm left"></div><div class="nx-robot-arm right"></div>
            <div class="nx-robot-torso"><div class="nx-robot-chest">{STAR_ICON}</div></div>
            <div class="nx-robot-leg left"></div><div class="nx-robot-leg right"></div>
          </div>
        </div>
        <span class="nx-float-chip chip-a">PDF</span><span class="nx-float-chip chip-b">DOCX</span><span class="nx-float-chip chip-c">RAG</span>
        <span class="nx-particle p-1"></span><span class="nx-particle p-2"></span>
        <span class="nx-particle p-3"></span><span class="nx-particle p-4"></span>
        <div class="nx-stage-label">LIVE VECTOR SEARCH · AI ASSISTED</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.markdown(
        f'<div class="nx-stat"><div class="nx-stat-label">Documents</div><div class="nx-stat-value">{len(st.session_state.file_names)}</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="nx-stat"><div class="nx-stat-label">Searchable chunks</div><div class="nx-stat-value">{st.session_state.indexed_chunks}</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="nx-stat"><div class="nx-stat-label">Retrieved per question</div><div class="nx-stat-value">{top_k}</div></div>',
        unsafe_allow_html=True,
    )

if st.session_state.file_names:
    cards = []
    for index, name in enumerate(st.session_state.file_names):
        safe_name = html.escape(str(name), quote=True)
        extension = str(name).rsplit(".", 1)[-1].upper() if "." in str(name) else "FILE"
        safe_extension = html.escape(extension[:8], quote=True)
        cards.append(
            f"""
            <article class="nx-file-card" role="listitem" style="--card-index:{index}">
              <div class="nx-file-card-inner">
                <div class="nx-file-card-top">
                  <span class="nx-file-glyph" aria-hidden="true"><svg viewBox="0 0 24 28" focusable="false"><path d="M4 1.5h10l6 6V25a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 25V1.5Z" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M14 1.5V8h6M8 14h8M8 18h8M8 22h5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></span>
                  <span class="nx-file-type">{safe_extension}</span>
                </div>
                <span class="nx-file-name" title="{safe_name}">{safe_name}</span>
                <span class="nx-file-meta">Indexed and ready to search</span>
              </div>
            </article>
            """
        )
    st.markdown(
        f"""
        <section class="nx-file-shelf" aria-label="Indexed files">
          <div class="nx-file-shelf-head">
            <div><div class="nx-eyebrow">Your knowledge base</div><h2 class="nx-file-shelf-title">Indexed files</h2></div>
            <span class="nx-file-count">{len(st.session_state.file_names)} files</span>
          </div>
          <div class="nx-file-grid" role="list">{''.join(cards)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="nx-empty-state">
          <div class="nx-empty-icon">↥</div>
          <div>
            <div class="nx-empty-title" style="font-weight:800;font-size:1.02rem">Your knowledge base is ready for its first document</div>
            <div class="nx-muted" style="margin-top:.25rem">Use the sidebar upload to add PDF, DOCX, CSV, JSON, TXT, or Markdown files. They will be indexed together for cross-document answers.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful document research assistant. Answer factual questions "
            "using only the supplied document context. Conversation history may help "
            "resolve references, but it is not evidence. If the documents do not contain "
            "the answer, say: \"I couldn't find that in the uploaded documents.\" "
            "Cite supporting sources using file name and page/row when available, "
            "for example (guide.pdf, p. 4). Never invent citations.\n\n"
            "Document context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ]
)

if st.session_state.vectorstore is None:
    st.stop()

with st.container(key="nx-chat-history"):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources used"):
                    for source in msg["sources"]:
                        st.markdown(f"**{source['file']} · {source['location']}**")
                        st.text(source["text"])
                        st.divider()

question = st.chat_input("Ask anything about your uploaded files…")

if question:
    prior = list(st.session_state.messages)
    chat_history = conversation_history(prior)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
        with st.chat_message("assistant"):
            with st.spinner("Searching your knowledge base..."):
                docs = retriever.invoke(question)
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=0,
                    api_key=st.session_state.api_key,
                )
                chain = RAG_PROMPT | llm | StrOutputParser()
                answer = chain.invoke(
                    {
                        "context": format_docs(docs),
                        "question": question,
                        "chat_history": chat_history,
                    }
                )
            st.markdown(answer)

            sources = [
                {
                    "file": str(doc.metadata.get("source", "Unknown file")),
                    "location": display_page(doc.metadata),
                    "text": doc.page_content[:900],
                }
                for doc in docs
            ]
            with st.expander("📚 Sources used"):
                for source in sources:
                    st.markdown(f"**{source['file']} · {source['location']}**")
                    st.text(source["text"])
                    st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
    except Exception as exc:
        st.error(
            "The question could not be answered. Check your API key, model access, "
            f"and network connection. Details: {exc}"
        )

st.markdown(
    '<p style="text-align:center;color:#7c879a;font-size:.66rem;padding:.35rem 0">NEXORA KNOWLEDGE STUDIO · Verify important answers against the cited source passages.</p>',
    unsafe_allow_html=True,
)

