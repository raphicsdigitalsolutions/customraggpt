import hashlib

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


# -------------------------- API key gate ----------------------------
# Use a reliable in-page gate instead of st.dialog so this works on
# older Streamlit versions as well and never leaves a blank page.

if not st.session_state.api_key:
    art_col, login_col = st.columns([1.05, 0.95], gap="large")

    with art_col:
        st.markdown(
            """
            <section class="nx-login-art">
              <div>
                <div class="nx-brand" style="border:0;margin:0;padding:0">
                  <span class="nx-mark">✦</span>
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
        '<div class="nx-brand"><span class="nx-mark">✦</span><span class="nx-brand-name">NEXORA</span><span class="nx-brand-accent">STUDIO</span></div>',
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
    """
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
        <div class="nx-hero-crystal"></div>
        <span class="nx-particle p-1"></span><span class="nx-particle p-2"></span>
        <span class="nx-particle p-3"></span><span class="nx-particle p-4"></span>
        <div class="nx-stage-label">LIVE VECTOR SEARCH · AI ASSISTED</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
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
    with st.expander("View indexed documents"):
        for name in st.session_state.file_names:
            st.markdown(f"📄 `{name}`")
else:
    st.markdown(
        """
        <div class="nx-empty-state">
          <div class="nx-empty-icon">↥</div>
          <div>
            <div style="font-weight:800;color:#f4f6ff;font-size:1.02rem">Your knowledge base is ready for its first document</div>
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
    '<p style="text-align:center;color:#7c879a;font-size:.76rem;padding:1.5rem 0">NEXORA KNOWLEDGE STUDIO · Verify important answers against the cited source passages.</p>',
    unsafe_allow_html=True,
)

