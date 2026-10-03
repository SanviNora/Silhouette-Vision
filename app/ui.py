"""Silhouette Vision — Streamlit app.

Run: streamlit run app/ui.py

Cover: a runway photo with a frosted-glass card (search by photo / by words). A photo upload is
"scanned" step by step with real progress (garment detection, attributes, luxury models, search,
exact match). Results below in an editorial dark layout; a product opens in a split detail view.
Also: style map, new-product demand forecast, about.
"""

import base64
import html
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # hosted: no pip install

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from silhouette_vision.config import DATA_REPO, DATA_ROOT, HOSTED, path
from silhouette_vision.images import load_rgb
from silhouette_vision.search import Filters, SearchEngine

st.set_page_config(page_title="Silhouette Vision", page_icon="👗", layout="wide",
                   initial_sidebar_state="expanded")
ss = st.session_state
ASSETS = Path(__file__).resolve().parent / "assets"
SOURCE_LABEL = {"zooclaw": "ZooClaw", "lookbench": "LookBench", "secondhand": "Second-hand", "abo": "Amazon"}
EXAMPLES = ["black leather bag", "white sneakers", "floral midi dress", "levi's jeans", "cream cardigan"]
COLOUR_HEX = {"Black": "#151515", "White": "#F5F3EE", "Off White": "#EEE8DC", "Cream": "#EFE6D2", "Grey": "#8E8E8E",
              "Grey Melange": "#A3A3A3", "Charcoal": "#3C3C3C", "Silver": "#C9C9C9", "Steel": "#7B8794",
              "Blue": "#3F6FC4", "Navy Blue": "#1F2A4D", "Turquoise Blue": "#3FB6B2", "Teal": "#2F7F7B",
              "Green": "#4E8A4B", "Olive": "#6E6B3A", "Red": "#C0392B", "Maroon": "#6D1F2A", "Pink": "#E59BB3",
              "Magenta": "#B3337F", "Peach": "#F1B79A", "Purple": "#6A4C93", "Lavender": "#B7A6D9",
              "Yellow": "#E8C63A", "Mustard": "#C9A227", "Orange": "#E07B39", "Rust": "#A9532F",
              "Brown": "#6B4A32", "Tan": "#B98A5E", "Beige": "#D8C3A0", "Khaki": "#B5A47A", "Gold": "#C8A24B",
              "Bronze": "#9C6B3C", "Copper": "#B0693E",
              "Multi": "conic-gradient(#E07B39,#3F6FC4,#4E8A4B,#E59BB3,#E07B39)"}
TIER_COLOUR = {"Very likely": "#9DB52F", "Possibly": "#C8916F", "No confident": "#8C877F"}
ABOUT = """
Portfolio project, not affiliated with any brand or retailer.
[Code, methods and evaluation](https://github.com/SanviNora/Silhouette-Vision)

**Products (2019–2026)**: *ZooClaw-Fashion* (2026, SerendipityOne, CC BY-NC 4.0) ·
*Second-Hand Fashion* (2022–24, Nauman et al., RISE, Wargön Innovation, Myrorna, CC BY 4.0,
doi:10.5281/zenodo.13788681) · *LookBench* studio gallery (2025, Apache-2.0) · footwear from
*Amazon Berkeley Objects* (c. 2019–21, Collins et al., CVPR 2022, CC BY 4.0).

**Demand**: *Visuelle 2.0*, Skenderi et al., CVPR Workshops 2022, CC BY-NC-SA 4.0.

**Models**: Marqo-FashionSigLIP, YOLOS-Fashionpedia; attribute heads trained on Fashion Product
Images (Myntra, MIT) and the catalog sources' own labels.
"""

# --- style --------------------------------------------------------------------------------------
GRAIN = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>"
         "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/>"
         "</filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>")
CLOUD = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 48' fill='none' "
         "stroke='white' stroke-opacity='.85' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'>"
         "<path d='M18 40H14a10 10 0 0 1-1.5-19.9A14 14 0 0 1 39.5 13 11 11 0 0 1 52 24a9 9 0 0 1-2 16h-4'/>"
         "<path d='M32 44V26M25 32l7-7 7 7'/></svg>")


@st.cache_data(show_spinner=False)
def css() -> str:
    hero = "data:image/jpeg;base64," + base64.b64encode((ASSETS / "runway.jpg").read_bytes()).decode()
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@300;400;500;600;700&family=Pinyon+Script&display=swap');
:root {{ --ink:#141312; --panel:#1E1D1B; --line:#34322F; --bone:#F1EEE7; --muted:#A39E95; --lime:#D8F36A; --rose:#E4B6A1; }}
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {{ font-family:'Inter',sans-serif; }}
[data-testid="stAppViewContainer"] {{ background: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stHeader"], [data-testid="stHeader"] div {{ pointer-events: none !important; }}
[data-testid="stHeader"] button, [data-testid="stHeader"] a {{ pointer-events: auto !important; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
[class*="st-key-hero"] > div:first-child, .st-key-herosm > div:first-child {{ padding-right: 40px; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid #DDD6C8; }}
section[data-testid="stSidebar"] h3 {{ font-family:'Anton',sans-serif; letter-spacing:.05em; font-weight:400; text-transform:uppercase; }}
footer {{ visibility:hidden; }}

/* hero: runway photo + film grain (cover page and section banners) */
.st-key-hero, .st-key-herosm {{ position:relative; padding: 18px 44px 64px; overflow:hidden;
  background: linear-gradient(180deg, rgba(8,8,8,.30) 0%, rgba(8,8,8,.12) 45%, rgba(20,19,18,.94) 100%),
              url('{hero}') center 42% / cover no-repeat; }}
.st-key-hero {{ min-height: 100vh; }}
.st-key-heroc {{ position:relative; padding: 18px 44px 46px; overflow:hidden;
  background: linear-gradient(180deg, rgba(8,8,8,.30) 0%, rgba(8,8,8,.12) 45%, rgba(20,19,18,.94) 100%),
              url('{hero}') center 42% / cover no-repeat; }}
.st-key-heroc::before {{ content:""; position:absolute; inset:0; background-image:url("{GRAIN}"); opacity:.16; mix-blend-mode:overlay; pointer-events:none; }}
.st-key-heroc > div {{ position:relative; z-index:1; }}
.st-key-heroc .st-key-glass {{ margin-top: 5vh; }}
.st-key-herosm {{ min-height: 340px; }}
.st-key-hero::before, .st-key-herosm::before {{ content:""; position:absolute; inset:0; background-image:url("{GRAIN}");
  opacity:.16; mix-blend-mode:overlay; pointer-events:none; }}
.st-key-hero > div, .st-key-herosm > div {{ position:relative; z-index:1; }}
.sv-logo {{ font-family:'Anton',sans-serif; font-size:1.7rem; letter-spacing:.06em; color:var(--bone); line-height:1; white-space:nowrap; padding-top:.25rem; }}
.sv-logo i {{ font-family:'Pinyon Script',cursive; font-style:normal; font-size:1.95rem; letter-spacing:0; color:var(--rose); margin-left:.3rem; }}
[class*="st-key-nav-"] button {{ background:transparent !important; border:none !important; color:var(--bone) !important; box-shadow:none !important;
  padding:.2rem .1rem !important; min-height:0 !important; }}
[class*="st-key-nav-"] button p {{ font-size:.76rem !important; letter-spacing:.16em; text-transform:uppercase; }}
[class*="st-key-nav-"] button:hover p, [class*="st-key-nav-"] button[data-testid="stBaseButton-primary"] p {{ color: var(--lime) !important; }}
.sv-hero-title {{ font-family:'Anton',sans-serif; color:var(--bone); font-size: clamp(3rem, 8vw, 7.4rem); line-height:.92;
  text-transform:uppercase; margin: 8vh 0 0 0; }}
.sv-hero-title em {{ font-style:normal; color: var(--lime); }}
.sv-hero-sub {{ color: rgba(241,238,231,.84); max-width: 640px; font-size:1rem; margin-top:.8rem; }}

/* frosted glass card (AI-Cloud concept, tinted as in the demo) */
.st-key-glass {{ margin-top: 10vh; border-radius: 28px; padding: 8px 16px 16px;
  background: linear-gradient(168deg, rgba(228,182,161,.32) 0%, rgba(62,58,60,.40) 40%, rgba(50,48,46,.44) 64%, rgba(154,154,74,.36) 100%);
  backdrop-filter: blur(26px) saturate(140%); -webkit-backdrop-filter: blur(26px) saturate(140%);
  border: 1px solid rgba(255,255,255,.22); box-shadow: 0 30px 90px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.28); }}
.st-key-glass [data-testid="stVerticalBlock"] {{ gap: .55rem; }}
.st-key-gbody {{ background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10); border-radius: 20px; padding: 16px 18px; }}
.sv-gtitle {{ text-align:center; color: rgba(255,255,255,.9); font-size:.92rem; font-weight:500; }}
.sv-gtitle span {{ font-family:'Pinyon Script',cursive; font-size:1.3rem; color:var(--rose); margin-right:.15rem; }}
.sv-gline {{ height:1px; background: rgba(255,255,255,.14); margin: 0 -16px 4px; }}
.st-key-gback button, .st-key-gclose button {{ width:30px; height:30px; min-height:0 !important; padding:0 !important; border-radius:50% !important;
  background: rgba(255,255,255,.14) !important; border: 1px solid rgba(255,255,255,.22) !important; color:#fff !important; }}
.st-key-gclose {{ display:flex; justify-content:flex-end; }}
.st-key-gophoto button, .st-key-gowords button {{ height:58px; border-radius:14px !important; background: rgba(250,248,244,.95) !important;
  border:none !important; }}
.st-key-gophoto button p, .st-key-gowords button p {{ font-size:1.08rem !important; font-weight:500; color: var(--ink); }}
.st-key-gophoto button p {{ color:#7E2F2F; }}
.st-key-gophoto button:hover, .st-key-gowords button:hover {{ background: var(--lime) !important; }}
.sv-hello {{ text-align:center; color: rgba(255,255,255,.86); font-size:.95rem; margin: .3rem 0 .9rem; }}
.sv-hello b {{ font-family:'Anton',sans-serif; font-weight:400; font-size:2rem; letter-spacing:.03em; display:block; color:#fff; margin-bottom:.15rem; }}
[class*="st-key-gact"] button {{ height:46px; border-radius:14px !important; background: rgba(255,255,255,.12) !important;
  border: 1px solid rgba(255,255,255,.16) !important; color:#fff !important; }}
[class*="st-key-gact"] button:hover {{ background: rgba(255,255,255,.22) !important; }}

/* drop zone (file uploader restyled) */
.st-key-glass [data-testid="stFileUploader"] label {{ display:none; }}
.st-key-glass [data-testid="stFileUploaderDropzone"] {{ background: transparent url("{CLOUD}") center 84px / 64px no-repeat !important;
  border: 1.5px dashed rgba(255,255,255,.30) !important; border-radius: 18px; min-height: 290px; padding: 182px 20px 26px !important;
  display:flex; flex-direction:column; align-items:center; justify-content:flex-end; position:relative; }}
.st-key-glass [data-testid="stFileUploaderDropzone"]::before {{ content: "Drop a photo here\\A to search 51,047 products"; white-space: pre;
.st-key-glass [class*="st-key-fc-up"] [data-testid="stFileUploaderDropzone"]::before {{ content: "Drop a product photo\\A to forecast its first 12 weeks"; }}
  position:absolute; top:22px; left:0; right:0; text-align:center; color: rgba(255,255,255,.92); font-size:1.15rem; line-height:1.55; font-weight:500; }}
.st-key-glass [data-testid="stFileUploaderDropzone"]::after {{ content:"or"; position:absolute; top:158px; left:0; right:0; text-align:center;
  color: rgba(255,255,255,.6); font-size:.85rem; }}
.st-key-glass [data-testid="stFileUploaderDropzoneInstructions"] {{ display:none !important; }}
.st-key-glass [data-testid="stFileUploaderDropzone"] button {{ font-size:0 !important; background: rgba(255,255,255,.14) !important;
  border:1px solid rgba(255,255,255,.2) !important; border-radius:12px !important; padding:.55rem 1.4rem !important; }}
.st-key-glass [data-testid="stFileUploaderDropzone"] button * {{ display:none !important; }}
.st-key-glass [data-testid="stFileUploaderDropzone"] button::after {{ content:"Select from your computer"; font-size:.92rem; color: rgba(255,255,255,.95); }}
.st-key-glass [data-testid="stFileUploaderFile"] {{ display:none; }}

/* write box */
.st-key-glass textarea {{ background: linear-gradient(160deg, rgba(214,160,140,.48), rgba(170,120,110,.38)) !important; color:#fff !important;
  font-size: 1.35rem !important; text-align:center; border-radius:16px !important; min-height:190px !important; padding-top:68px !important; }}
.st-key-glass textarea::placeholder {{ color: rgba(255,255,255,.85); }}
.st-key-glass [data-baseweb="textarea"] {{ background: transparent !important; border:1px solid rgba(255,255,255,.18) !important; border-radius:16px !important; }}
.st-key-glass [data-testid="stTextArea"] label, .st-key-glass [data-testid="stTextInput"] label {{ display:none; }}
.st-key-glass input {{ background: rgba(255,255,255,.10) !important; color:#fff !important; }}
.st-key-glass [data-baseweb="input"] {{ background: transparent !important; border:1px solid rgba(255,255,255,.18) !important; border-radius:12px !important; }}
.st-key-exrow {{ flex-wrap: wrap; justify-content:center; }}
.st-key-exrow button {{ background: rgba(255,255,255,.10) !important; border:1px solid rgba(255,255,255,.16) !important;
  border-radius:999px !important; min-height:0 !important; padding:.3rem .35rem !important; }}
.st-key-exrow button p {{ font-size:.74rem !important; color: rgba(255,255,255,.92); white-space:nowrap; }}

/* scanning (real progress) */
.sv-scan {{ position:relative; border-radius:16px; overflow:hidden; background: rgba(0,0,0,.25); height:210px; display:flex;
  align-items:center; justify-content:center; margin-bottom:.4rem; }}
.sv-scan img {{ max-height:210px; max-width:100%; object-fit:contain; }}
.sv-scan::after {{ content:""; position:absolute; left:0; right:0; height:70px; top:-70px;
  background: linear-gradient(180deg, rgba(216,243,106,0), rgba(216,243,106,.42) 85%, rgba(255,255,255,.95));
  animation: scan 1.6s cubic-bezier(.4,0,.2,1) infinite; }}
@keyframes scan {{ 0% {{ top:-70px; }} 100% {{ top:100%; }} }}
.sv-row {{ display:flex; align-items:center; gap:.8rem; padding:.5rem .2rem; border-bottom:1px solid rgba(255,255,255,.08); color:#fff; }}
.sv-row:last-child {{ border-bottom:none; }}
.sv-ico {{ width:34px; height:34px; border-radius:9px; background: linear-gradient(160deg,#6FD3F7,#2A9FD6); flex:none;
  display:flex; align-items:center; justify-content:center; font-size:.95rem; }}
.sv-row .t {{ flex:1; font-size:.92rem; line-height:1.2; }}
.sv-row .t small {{ display:block; color: rgba(255,255,255,.55); font-size:.74rem; margin-top:.15rem; }}
.sv-pill {{ --p:0; width:54px; height:26px; border-radius:999px; display:flex; align-items:center; justify-content:center; font-size:.78rem;
  color:#fff; border:1.5px solid rgba(255,255,255,.3); background: linear-gradient(90deg, rgba(255,255,255,.24) calc(var(--p) * 1%), rgba(255,255,255,.05) 0); }}
.sv-pill.run {{ border-color: var(--lime); }}
.sv-pill.ok {{ border-color: rgba(255,255,255,.8); }}
.sv-progpanel {{ margin-top:16px; border-radius:22px; padding:16px 20px; color:#fff;
  background: linear-gradient(168deg, rgba(110,104,70,.58), rgba(66,62,50,.58)); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
  border:1px solid rgba(255,255,255,.18); }}
.sv-prog-top, .sv-prog-bot {{ display:flex; justify-content:space-between; font-size:.88rem; }}
.sv-prog-bot {{ color: rgba(255,255,255,.65); font-size:.8rem; }}
.sv-bar {{ height:4px; border-radius:999px; background: rgba(255,255,255,.18); margin:.7rem 0 .6rem; overflow:hidden; }}
.sv-bar > div {{ height:100%; background:#fff; border-radius:999px; }}
.sv-spin {{ display:inline-block; width:12px; height:12px; border:2px solid rgba(255,255,255,.35); border-top-color:#fff; border-radius:50%;
  margin-right:.45rem; vertical-align:-1px; animation: spin .8s linear infinite; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.st-key-gbody [data-testid="stImage"] img {{ border-radius:12px; max-height:150px; object-fit:contain; }}
.st-key-gbody [data-testid="stImageCaption"], .st-key-gbody [data-testid="stCaptionContainer"] {{ color: rgba(255,255,255,.75) !important; }}
.st-key-gbody [data-testid="stRadio"] p {{ color:#fff !important; }}

/* editorial sections (Auralee) */
.st-key-results, .st-key-section {{ padding: 26px 56px 70px; background: var(--ink); }}
.sv-marquee {{ overflow:hidden; white-space:nowrap; border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:.7rem 0; margin: 0 -56px 2rem; }}
.sv-marquee span {{ display:inline-block; font-family:'Anton',sans-serif; font-size:2.1rem; color: var(--bone); letter-spacing:.04em;
  animation: marquee 34s linear infinite; padding-right:2rem; }}
.sv-marquee span b {{ color: var(--lime); font-weight:400; }}
@keyframes marquee {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-100%); }} }}
.sv-paper {{ display:inline-block; background:#EFEBE3; color: var(--ink); font-weight:600; font-size:.95rem; padding:.35rem 1rem .4rem; margin:.4rem 0 .9rem;
  transform: rotate(-1.6deg); box-shadow: 0 6px 18px rgba(0,0,0,.35);
  clip-path: polygon(0 8%, 6% 0, 14% 6%, 23% 1%, 33% 7%, 44% 0, 55% 6%, 66% 1%, 77% 7%, 88% 0, 100% 6%, 98% 52%, 100% 94%, 90% 100%, 79% 94%, 68% 100%, 57% 95%, 45% 100%, 33% 94%, 22% 100%, 11% 95%, 0 100%, 2% 50%); }}
.sv-display {{ font-family:'Anton',sans-serif; text-transform:uppercase; color: var(--bone); font-size: clamp(2.4rem, 5vw, 4.4rem); line-height:.95; margin:.2rem 0 .6rem; }}
.sv-display em {{ font-style:normal; color: var(--lime); }}
.sv-note {{ color: var(--muted); font-size:.86rem; max-width:780px; margin-bottom:1rem; }}
.sv-chip {{ display:inline-block; padding:.22rem .7rem; margin:0 .35rem .4rem 0; border-radius:999px; border:1px solid var(--line);
  color: var(--bone); font-size:.78rem; background: var(--panel); }}
.sv-chip.same {{ border-color:#5D6B2A; color: var(--lime); }}
.sv-chip.similar {{ border-color:#6F5546; color: var(--rose); }}
.sv-chip.different {{ color: var(--muted); }}
.sv-chip.lime {{ background: var(--lime); color: var(--ink); border-color: var(--lime); font-weight:600; }}

/* product cards (Auralee) */
.av-card {{ background: var(--panel); border:1px solid var(--line); }}
.av-top {{ padding:12px 14px 10px; min-height:84px; }}
.av-brand {{ font-family:'Anton',sans-serif; color: var(--lime); font-size:.95rem; letter-spacing:.03em; text-transform:uppercase; min-height:1.15rem; }}
.av-title {{ font-family:'Anton',sans-serif; color: var(--bone); font-size:1.05rem; line-height:1.08; text-transform:uppercase;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
.av-img {{ background:#ECE9E3; height:290px; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; }}
.av-img img {{ max-height:272px; max-width:90%; object-fit:contain; filter: grayscale(1) contrast(1.04); transition: filter .45s ease, transform .45s ease; }}
.av-card:hover .av-img img {{ filter:none; transform: scale(1.035); }}
.av-badge {{ position:absolute; top:10px; right:10px; background: var(--ink); color: var(--bone); font-size:.64rem; letter-spacing:.1em;
  text-transform:uppercase; padding:.18rem .5rem; }}
.av-tags {{ padding:10px 14px 0; height:2.1rem; overflow:hidden; }}
.av-tags .sv-chip {{ font-size:.68rem; padding:.1rem .45rem; }}
.av-bottom {{ display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border-top:1px solid var(--line); color: var(--muted); font-size:.76rem; }}
.av-dots span {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; border:1px solid rgba(255,255,255,.35); }}
[class*="st-key-view-"] button {{ background: transparent !important; border:1px solid var(--line) !important; border-top:none !important;
  border-radius:0 !important; margin-top:-1rem; }}
[class*="st-key-view-"] button p {{ font-size:.72rem !important; letter-spacing:.18em; text-transform:uppercase; color: var(--bone); }}
[class*="st-key-view-"] button:hover {{ border-color: var(--lime) !important; }}
[class*="st-key-view-"] button:hover p {{ color: var(--lime); }}

/* best match */
.bm {{ background:#EFEBE3; color: var(--ink); padding:26px 28px; min-height: 452px; }}
.st-key-gbody [data-testid="stElementContainer"]:has([data-testid="stImage"]) {{ width:100% !important; display:flex; justify-content:center; }}
.bm .eb {{ font-size:.7rem; letter-spacing:.18em; text-transform:uppercase; color:#7A746B; }}
.bm h3 {{ font-family:'Anton',sans-serif; font-weight:400; text-transform:uppercase; font-size:2.3rem; line-height:1; margin:.35rem 0 .9rem; color: var(--ink); }}
.bm .bar {{ height:10px; background:#DCD5C8; }}
.bm .bar > div {{ height:100%; }}
.bm .row {{ display:flex; justify-content:space-between; margin:.55rem 0 .9rem; font-size:.9rem; }}
.bm p {{ color:#5A544C; font-size:.84rem; line-height:1.55; }}

/* product detail (Allure) */
.st-key-detail {{ background:#F4F1EA; }}
.dt-img {{ background:#E9E5DD; height:620px; display:flex; align-items:center; justify-content:center; }}
.dt-img img {{ max-height:570px; max-width:86%; object-fit:contain; }}
.dt-thumbs {{ display:flex; gap:10px; padding:12px 26px 22px; background:#E9E5DD; }}
.dt-thumbs div {{ flex:1; background:#fff; height:110px; display:flex; align-items:center; justify-content:center; border:1px solid #DDD6CA; position:relative; }}
.dt-thumbs img {{ max-height:96px; max-width:92%; object-fit:contain; }}
.dt-thumbs span {{ position:absolute; bottom:3px; left:6px; font-size:.58rem; letter-spacing:.1em; text-transform:uppercase; color:#7A746B; }}
.dt-panel {{ background:#fff; color: var(--ink); padding:34px 38px 28px; margin:26px 26px 0 0; }}
.dt-crumb {{ font-size:.76rem; letter-spacing:.06em; text-transform:uppercase; color:#3A3632; }}
.dt-crumb b {{ font-weight:400; color:#A39E95; margin:0 .55rem; }}
.dt-title {{ font-weight:600; text-transform:uppercase; letter-spacing:-.02em; font-size:2.3rem; line-height:1.04; margin:1.1rem 0 .6rem; color: var(--ink); }}
.dt-price {{ font-size:2rem; color:#B5745A; font-weight:500; margin-bottom:1.4rem; }}
.dt-grid {{ display:grid; grid-template-columns: 34% 66%; row-gap:1.05rem; font-size:.84rem; }}
.dt-grid .k {{ color:#A39E95; text-transform:uppercase; letter-spacing:.06em; font-size:.76rem; padding-top:.15rem; }}
.dt-grid .v {{ color: var(--ink); text-transform:uppercase; }}
.dt-grid .sz {{ display:inline-block; padding:.12rem .5rem; background:#C8916F; color:#fff; }}
.dt-grid ul {{ margin:0; padding-left:1rem; }}
.dt-grid li::marker {{ color:#C8916F; }}
.dt-cta {{ display:block; background: var(--ink); color: var(--bone) !important; text-align:center; padding:1.05rem; margin:0 26px 0 0;
  letter-spacing:.2em; text-transform:uppercase; font-size:.8rem; text-decoration:none !important; }}
.dt-cta:hover {{ background: var(--lime); color: var(--ink) !important; }}
.st-key-dback button {{ background: transparent !important; border:1px solid #CFC7B8 !important; border-radius:999px !important; margin-top:.8rem; }}
.st-key-dback button p, .st-key-heatb button p {{ color: var(--ink) !important; }}
.st-key-heatb button {{ background:#F4F1EA !important; border:1px solid #CFC7B8 !important; }}
.st-key-detail [data-testid="stExpander"] {{ background:#fff; border:1px solid #E1DACC; margin-right:26px; }}
.st-key-detail [data-testid="stExpander"] summary p, .st-key-detail [data-testid="stCaptionContainer"] p {{ color: var(--ink) !important; }}
.st-key-detail [data-testid="stExpander"] summary {{ background:#fff !important; color: var(--ink) !important; }}
.st-key-detail [data-testid="stExpander"] summary svg {{ fill: var(--ink); color: var(--ink); }}
.st-key-detail .sv-chip {{ background:#F4F1EA; color: var(--ink); border-color:#DDD6CA; }}
.st-key-detail .sv-chip.same {{ background:#EEF6D2; border-color:#C9DD7A; color:#3D4A12; }}
.st-key-detail .sv-chip.similar {{ background:#F7E7DF; border-color:#E9C9B8; color:#7B4A35; }}

/* metrics */
[data-testid="stMetricValue"] {{ font-family:'Anton',sans-serif; font-size:2.3rem; color: var(--bone); }}
[data-testid="stMetricLabel"] p {{ color: var(--muted) !important; letter-spacing:.06em; text-transform:uppercase; font-size:.72rem !important; }}
</style>"""


# --- cached resources -------------------------------------------------------------------------
def release_memory() -> None:
    """Return freed memory to the OS after heavy steps (Linux keeps it otherwise; the free host
    has ~2.7 GB). No-op elsewhere."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


@st.cache_resource(show_spinner="First start: downloading the catalog (~1–2 minutes)…")
def get_data() -> None:
    if HOSTED:
        from silhouette_vision.bootstrap import ensure_bundle

        ensure_bundle(DATA_ROOT, DATA_REPO)
        release_memory()


@st.cache_resource(show_spinner="Loading the search model…")
def get_engine() -> SearchEngine:
    get_data()
    return SearchEngine("marqo_fashion_siglip")


@st.cache_resource(show_spinner=False)
def get_detector():
    from silhouette_vision.detect import GarmentDetector

    return GarmentDetector()


@st.cache_resource(show_spinner=False)
def get_attributes():
    try:
        from silhouette_vision.enrich import AttributePredictor, load_catalog_predictions

        return AttributePredictor(), load_catalog_predictions()
    except FileNotFoundError:
        return None, None


@st.cache_resource(show_spinner=False)
def get_match_confidence():
    try:
        from silhouette_vision.match import MatchConfidence

        return MatchConfidence()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def get_recogniser():
    try:
        from silhouette_vision.named_models import ModelRecogniser

        return ModelRecogniser(get_engine().text_vector)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading the demand model…")
def get_forecaster():
    try:
        from silhouette_vision.demand import DemandForecaster

        return DemandForecaster()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def get_style_map():
    file = DATA_ROOT / "data/processed/style_clusters.parquet"
    report = path("reports") / "style_clusters.json"
    if not file.exists() or not report.exists():
        return None, None
    return pd.read_parquet(file), json.loads(report.read_text())


@st.cache_data(show_spinner=False, max_entries=300)
def data_uri(file: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(file).read_bytes()).decode()


def pil_uri(img: Image.Image, side: int = 640) -> str:
    img = img.convert("RGB").copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --- small helpers ------------------------------------------------------------------------------
RELATION_MARK = {"same": "✓", "similar": "≈", "different": "✗"}


def esc(v) -> str:
    return html.escape(str(v)) if isinstance(v, str) else ""


def chips(values, kind: str = "") -> str:
    return "".join(f'<span class="sv-chip {kind}">{esc(v)}</span>' for v in values if v)


def price_text(row) -> str:
    band = getattr(row, "price_band", None)
    return f"Resale {band} SEK" if isinstance(band, str) and band else ""


def go(view: str) -> None:
    ss.view, ss.detail = view, None
    st.rerun()


def item_preds(item_id):
    _, preds = get_attributes()
    return None if preds is None or item_id not in preds.index else preds.loc[item_id]


def card_tags(row, query_attrs) -> list[str]:
    p = item_preds(row.item_id)
    if p is None:
        return []
    from silhouette_vision.enrich import item_tags
    from silhouette_vision.explain import compare_attributes

    if query_attrs:
        comp = compare_attributes(query_attrs, p)
        shared = [f"{RELATION_MARK[c['relation']]} {c['item']}" for c in comp if c["relation"] != "different"]
        if shared:
            return shared[:3]
    return item_tags(p)


def colour_dots(row) -> str:
    p = item_preds(row.item_id)
    colour = p.get("pred_colour") if p is not None else None
    swatch = COLOUR_HEX.get(colour, "#555") if isinstance(colour, str) else "#555"
    return (f'<span class="av-dots" title="{esc(colour)}"><span style="background:{swatch}"></span>'
            '<span></span><span></span></span>')


def card(row, tags, right: str) -> str:
    badge = "pre-owned" if row.is_preowned else SOURCE_LABEL.get(row.source, row.source)
    return (f'<div class="av-card"><div class="av-top"><div class="av-brand">{esc(row.brand) or "&nbsp;"}</div>'
            f'<div class="av-title">{esc(row.title)}</div></div>'
            f'<div class="av-img"><img src="{data_uri(str(DATA_ROOT / row.image_path))}"/><span class="av-badge">{esc(badge)}</span></div>'
            f'<div class="av-tags">{chips(tags)}</div>'
            f'<div class="av-bottom">{colour_dots(row)}<span>{esc(right)}</span></div></div>')


def open_detail(item_id: str, origin: str) -> None:
    ss.detail, ss.detail_from = item_id, origin
    st.rerun()


def grid(results: pd.DataFrame, prefix: str, query_attrs=None, show_similarity=True) -> None:
    if results.empty:
        st.info("No products match these filters.")
        return
    cols = st.columns(4, gap="medium")
    for i, row in results.reset_index(drop=True).iterrows():
        with cols[i % 4]:
            sim = f"{row.similarity:.2f} match" if show_similarity and row.similarity == row.similarity else ""
            right = " · ".join(t for t in [price_text(row), sim] if t)
            st.markdown(card(row, card_tags(row, query_attrs), right), unsafe_allow_html=True)
            if st.button("[ View ]", key=f"view-{prefix}-{row.item_id}", width="stretch"):
                open_detail(row.item_id, prefix)


# --- sidebar ------------------------------------------------------------------------------------
def sidebar_filters(engine: SearchEngine) -> tuple[Filters, int, bool]:
    cat = engine.catalog
    st.sidebar.markdown("### Filters")
    sources = st.sidebar.multiselect("Catalog", list(SOURCE_LABEL), format_func=SOURCE_LABEL.get,
                                     help="Second-hand garments are 31.9k of the 51k products; pick sources to narrow down.")
    categories = st.sidebar.multiselect("Category", sorted(cat.category.unique()), format_func=str.capitalize)
    genders = st.sidebar.multiselect("For", sorted(cat.gender.dropna().unique()))
    condition = st.sidebar.radio("Condition", ["Any", "New", "Pre-owned"], horizontal=True)
    k = st.sidebar.slider("Results", 4, 48, 12, step=4)
    match_colour = st.sidebar.toggle(
        "Match colour", value=engine.has_colour, disabled=not engine.has_colour,
        help="Re-ranks photo results toward the photo's colours: colour agreement 50.8% → 54.8% on held-out garments.")
    st.sidebar.caption(f"{len(cat):,} products, 2019–2026 · search model: Marqo-FashionSigLIP")
    with st.sidebar.expander("About & credits"):
        st.markdown(ABOUT)
    preowned = {"Any": None, "New": False, "Pre-owned": True}[condition]
    return Filters(sources, categories, genders, preowned), k, match_colour


# --- hero pieces --------------------------------------------------------------------------------
NAV = [("search", "[ Search ]"), ("styles", "[ Style map ]"), ("forecast", "[ Forecast ]"), ("about", "[ About ]")]


def nav_bar() -> None:
    logo, *cols = st.columns([5, 1, 1.25, 1.15, 1], vertical_alignment="center")
    logo.markdown('<div class="sv-logo">SILHOUETTE<i>vision</i></div>', unsafe_allow_html=True)
    current = "search" if ss.view in ("home", "photo", "words") else ss.view
    for col, (view, label) in zip(cols, NAV, strict=True):
        with col, st.container(key=f"nav-{view}"):
            if st.button(label, key=f"navb-{view}", type="primary" if view == current and not ss.detail else "secondary"):
                reset_photo()
                go("home" if view == "search" else view)


def reset_photo() -> None:
    ss.pop("analysis", None)
    ss.pop("refine", None)
    ss.upload_n = ss.get("upload_n", 0) + 1


def glass_header(title_html: str, back: bool = True, close: bool = True) -> None:
    a, b, c = st.columns([1, 6, 1], vertical_alignment="center")
    with a:
        if back and st.button("‹", key="gback"):
            reset_photo()
            go("home")
    b.markdown(f'<div class="sv-gtitle">{title_html}</div>', unsafe_allow_html=True)
    with c:
        if close and st.button("✕", key="gclose"):
            reset_photo()
            go("home")
    st.markdown('<div class="sv-gline"></div>', unsafe_allow_html=True)


TITLE = "<span>Silhouette</span> Vision"


# --- photo analysis: the scan, with real progress -----------------------------------------------
STEPS = [("🔎", "Detecting garments", "finding the items in your photo"),
         ("🧵", "Reading the look", "type, colour, pattern, sleeves, neckline"),
         ("👜", "Recognising luxury models", "161 iconic models, by name"),
         ("✦", "Searching 51,047 products", "visual similarity, colour-aware"),
         ("◎", "Checking for an exact match", "calibrated likelihood")]


def scan_rows(done: int, running: int | None) -> str:
    rows = []
    for i, (icon, name, sub) in enumerate(STEPS):
        p = 100 if i < done else (50 if i == running else 0)
        cls = "ok" if i < done else ("run" if i == running else "")
        rows.append(f'<div class="sv-row"><div class="sv-ico">{icon}</div><div class="t">{name}<small>{sub}</small></div>'
                    f'<div class="sv-pill {cls}" style="--p:{p}">{p}</div></div>')
    return "".join(rows)


def progress_panel(done: int, elapsed: float, finished: float | None = None) -> str:
    n = len(STEPS)
    pct = round(100 * done / n)
    if finished is not None:
        head, right = f"Analysed {n}/{n}", f"{finished:.1f} sec"
    else:
        head = f'<span class="sv-spin"></span>Analysing {min(done + 1, n)}/{n}'
        right = f"~{max(1, round(elapsed / max(done, 1) * (n - done)))} sec left" if done else "starting…"
    return (f'<div class="sv-progpanel"><div class="sv-prog-top"><span>{head}</span><span>{pct}%</span></div>'
            f'<div class="sv-bar"><div style="width:{pct}%"></div></div>'
            f'<div class="sv-prog-bot"><span>{done} of {n} steps · 51,047 products</span><span>{right}</span></div></div>')


def analyse(image: Image.Image, crops: list, choice: int, engine: SearchEngine) -> dict:
    """Embedding, attributes and named model for the chosen garment (no animation)."""
    query_img = image if choice == 0 else crops[choice - 1]
    vec = engine.image_vector(query_img)
    predictor, _ = get_attributes()
    recogniser = get_recogniser()
    return {"query_img": query_img, "vec": vec, "choice": choice,
            "attrs": predictor.predict(vec) if predictor is not None else [],
            "named": recogniser.recognise(vec)[0] if recogniser is not None else None}


def scan(upload, engine: SearchEngine, body, prog) -> None:
    """Run the analysis step by step; each row and the progress panel update as a step finishes."""
    image = load_rgb(upload)
    uri = pil_uri(image)
    t0 = time.time()

    def step(i, fn):
        body.markdown(f'<div class="sv-scan"><img src="{uri}"/></div>' + scan_rows(i, i), unsafe_allow_html=True)
        prog.markdown(progress_panel(i, time.time() - t0), unsafe_allow_html=True)
        start = time.time()
        out = fn()
        time.sleep(max(0.0, 0.45 - (time.time() - start)))  # long enough to read each step
        return out

    detections = step(0, lambda: get_detector().items(image))
    crops = [d.crop(image) for d in detections[:5]]
    choice = 1 if len(crops) == 1 else 0  # several items: start from the whole photo
    query_img = image if choice == 0 else crops[choice - 1]
    predictor, _ = get_attributes()

    def read_look():
        v = engine.image_vector(query_img)
        return v, (predictor.predict(v) if predictor is not None else [])

    vec, attrs = step(1, read_look)
    recogniser = get_recogniser()
    named = step(2, lambda: recogniser.recognise(vec)[0] if recogniser is not None else None)
    step(3, lambda: engine.search(vec, 12))
    step(4, lambda: engine.best_match(vec))
    ss.analysis = {"image": image, "crops": crops, "seconds": time.time() - t0, "query_img": query_img,
                   "vec": vec, "attrs": attrs, "named": named, "choice": choice}
    body.markdown(f'<div class="sv-scan"><img src="{uri}"/></div>' + scan_rows(len(STEPS), None), unsafe_allow_html=True)
    prog.markdown(progress_panel(len(STEPS), 0, ss.analysis["seconds"]), unsafe_allow_html=True)
    release_memory()
    time.sleep(0.5)
    st.rerun()


# --- search page --------------------------------------------------------------------------------
def search_page(engine, filters, k, match_colour) -> None:
    view = ss.view
    if "pending_q" in ss:  # an example chip was clicked: fill the box before it is drawn
        ss.words_q = ss.pop("pending_q")
    has_results = (view == "photo" and ss.get("analysis")) or (view == "words" and ss.get("words_q", "").strip())
    with st.container(key="heroc" if has_results else "hero"):
        nav_bar()
        with st.columns([1, 1.35, 1])[1]:
            glass = st.container(key="glass")
            prog = st.empty()  # the progress panel sits under the card
            with glass:
                if view == "home":
                    glass_header(TITLE, back=False, close=False)
                    with st.container(key="gbody"):
                        st.markdown('<div class="sv-hello"><b>FIND THE LOOK</b>Search 51,047 products from 2019–2026 '
                                    'by photo or by words</div>', unsafe_allow_html=True)
                        if st.button("Search by photo", key="gophoto", width="stretch"):
                            go("photo")
                        if st.button("Search by words", key="gowords", width="stretch"):
                            go("words")
                elif view == "photo":
                    photo_glass(engine, prog)
                else:
                    words_glass()
            if view == "photo" and ss.get("analysis"):
                prog.markdown(progress_panel(len(STEPS), 0, ss.analysis["seconds"]), unsafe_allow_html=True)
    if view == "photo" and ss.get("analysis"):
        photo_results(engine, filters, k, match_colour)
    elif view == "words" and ss.get("words_q", "").strip():
        words_results(engine, filters, k)


def photo_glass(engine, prog) -> None:
    glass_header(TITLE + " · photo")
    a = ss.get("analysis")
    with st.container(key="gbody"):
        if a is None:
            if ss.get("pending_upload") is not None:  # a photo just arrived: the card becomes the scan
                scan(io.BytesIO(ss.pop("pending_upload")), engine, st.empty(), prog)
                return
            upload = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"], key=f"up-{ss.get('upload_n', 0)}")
            if upload is not None:
                ss.pending_upload = upload.getvalue()
                ss.upload_n = ss.get("upload_n", 0) + 1
                st.rerun()
            return
        options = ["Whole photo"] + [f"Item {i + 1}" for i in range(len(a["crops"]))]
        cols = st.columns(max(len(options), 3))
        for col, img, label in zip(cols, [a["image"], *a["crops"]], options, strict=False):
            col.image(img, caption=label, width="stretch")
        if len(options) > 1:
            choice = st.radio("Search for", options, index=a["choice"], horizontal=True, key="garment")
            idx = options.index(choice)
            if idx != a["choice"]:
                ss.analysis.update(analyse(a["image"], a["crops"], idx, engine))
                st.rerun()
        st.text_input("Refine", key="refine", placeholder="Refine with words: in red, leather, cropped…")
    c1, c2 = st.columns(2)
    with c1, st.container(key="gact-cancel"):
        if st.button("Cancel", key="gact-cancel-b", width="stretch"):
            reset_photo()
            go("home")
    with c2, st.container(key="gact-new"):
        if st.button("New photo", key="gact-new-b", width="stretch"):
            reset_photo()
            st.rerun()


def words_glass() -> None:
    glass_header(TITLE + " · words")
    with st.container(key="gbody"):
        st.text_area("Describe it", key="words_q", placeholder="write here…", height=190)
        with st.container(key="exrow", horizontal=True, gap="small"):
            for i, ex in enumerate(EXAMPLES):
                if st.button(ex, key=f"exb-{i}"):
                    ss.pending_q = ex
                    st.rerun()
    with st.container(key="gact-search"):
        st.button("Search  ↵", key="gact-search-b", width="stretch")


def marquee(words: list[str]) -> str:
    text = "".join(f"{w} <b>·</b> " for w in words) * 8
    return f'<div class="sv-marquee"><span>{text}</span><span>{text}</span></div>'


def photo_results(engine, filters, k, match_colour) -> None:
    a = ss.analysis
    query = a["vec"]
    if ss.get("refine"):
        query = engine.combine(query, engine.text_vector(ss.refine), 0.3)
    with st.container(key="results"):
        st.markdown(marquee(["FIND", "MATCH", "UNDERSTAND"]), unsafe_allow_html=True)
        left, right = st.columns([5, 7], gap="large")
        with left:
            st.markdown('<span class="sv-paper">what we see</span>', unsafe_allow_html=True)
            st.markdown(chips([f"{x['label']}: {x['value']} · {x['confidence']:.0%}" for x in a["attrs"]]),
                        unsafe_allow_html=True)
            predictor, _ = get_attributes()
            kind = next((x["value"] for x in a["attrs"] if x["attribute"] == "article_type"), None)
            if predictor is not None and predictor.type_to_category.get(kind) == "jewellery":
                st.info("This looks like jewellery, which the catalog doesn't carry: results show the closest accessories.")
            named_panel(engine, a)
        with right:
            best_match_panel(engine, query, filters, a)
        colour = engine.image_colour(a["query_img"]) if match_colour else None
        st.markdown('<div class="sv-display" style="margin-top:2.2rem">Your <em>look-alikes</em></div>'
                    '<div class="sv-note">✓ shared and ≈ close attributes with your photo; the number is visual similarity '
                    '(1.00 = identical image). Hover a card to see its colour.</div>', unsafe_allow_html=True)
        grid(engine.search(query, k, filters, query_colour=colour), "img", a["attrs"])


def named_panel(engine, a) -> None:
    named = a.get("named")
    if not named or named[1] < 0.5:
        return
    from urllib.parse import quote_plus

    from silhouette_vision.named_models import catalog_listings

    model, prob = named
    rows = catalog_listings(engine.catalog, model)
    web = f"https://www.google.com/search?tbm=shop&q={quote_plus(model.label)}"
    stock = f"{len(rows)} listing{'s' if len(rows) != 1 else ''} in the catalog" if len(rows) else "Not in our catalog"
    st.markdown(f'<span class="sv-paper">{"recognised model" if prob >= 0.8 else "probably"}</span>'
                f'<div class="sv-display" style="font-size:2.6rem">{esc(model.label)}</div>'
                f'{chips([f"model confidence {prob:.0%}"], "lime")}'
                f'<div class="sv-note">{esc(stock)} · <a href="{web}" target="_blank" style="color:var(--rose)">find it online ↗</a></div>',
                unsafe_allow_html=True)
    if len(rows):
        best = rows[np.argsort(-engine.similarities(a["vec"], rows))[:3]]
        cols = st.columns(3)
        for col, (_, row) in zip(cols, engine.catalog.iloc[best].iterrows(), strict=False):
            col.markdown(f'<div class="av-img" style="height:150px"><img style="max-height:136px" '
                         f'src="{data_uri(str(DATA_ROOT / row.image_path))}"/></div>', unsafe_allow_html=True)


def best_match_panel(engine, query, filters, a) -> None:
    matcher = get_match_confidence()
    if matcher is None:
        return
    row, top = engine.best_match(query, filters)
    prob = matcher.probability(top)
    tier = matcher.tier(prob)
    colour = next(c for key, c in TIER_COLOUR.items() if tier.startswith(key))
    c1, c2 = st.columns([1, 1.25], gap="small")
    with c1:
        st.markdown(card(row, card_tags(row, a["attrs"]), f"{row.similarity:.2f} match"), unsafe_allow_html=True)
        if st.button("[ View ]", key=f"view-best-{row.item_id}", width="stretch"):
            open_detail(row.item_id, "img")
    with c2:
        st.markdown(f'<div class="bm"><div class="eb">best match</div><h3>{esc(tier)}</h3>'
                    f'<div class="bar"><div style="width:{prob:.0%};background:{colour}"></div></div>'
                    f'<div class="row"><b>Exact-match likelihood {prob:.0%}</b><span>similarity {row.similarity:.2f}</span></div>'
                    "<p>Calibrated on benchmark photos: when this says 80%+, it was the exact product 88% of the time; "
                    "for products we don't stock it claims 80%+ for 0.1% of photos. It rewards a result that stands out "
                    "from the rest, not raw similarity. Open the product to see why it matches.</p></div>",
                    unsafe_allow_html=True)


def words_results(engine, filters, k) -> None:
    text = ss.words_q.strip()
    with st.container(key="results"):
        st.markdown(marquee(["DESCRIBE", "FIND", "COMPARE"]), unsafe_allow_html=True)
        st.markdown(f'<span class="sv-paper">results for</span><div class="sv-display">“{esc(text)}”</div>'
                    '<div class="sv-note">Matched on look (image embedding) and words (brand, title, type) together.</div>',
                    unsafe_allow_html=True)
        # No similarity number: text-image cosines are small by nature (~0.1) and would read as poor.
        grid(engine.search(engine.text_vector(text), k, filters, keywords=text), "txt", show_similarity=False)


# --- product detail (Allure) ----------------------------------------------------------------------
def detail_page(engine, filters, k, match_colour) -> None:
    from urllib.parse import quote_plus

    item_id = ss.detail
    row = engine.catalog.iloc[engine._row(item_id)]
    a = ss.get("analysis") if ss.get("detail_from") == "img" else None
    p = item_preds(item_id)
    heat = ss.get("heat", {}).get(item_id)
    with st.container(key="herosm"):
        nav_bar()
        st.markdown('<div class="sv-hero-title" style="font-size:clamp(2.4rem,5vw,4.2rem)">The <em>piece</em></div>',
                    unsafe_allow_html=True)
    with st.container(key="detail"):
        left, right = st.columns([7, 5], gap="small")
        thumbs = [("product", data_uri(str(DATA_ROOT / row.image_path)))]
        if a is not None:
            thumbs.append(("your photo", pil_uri(a["query_img"], 300)))
        if heat:
            thumbs += [("heat · yours", heat[0]), ("heat · match", heat[1])]
        left.markdown(f'<div class="dt-img"><img src="{thumbs[0][1]}"/></div><div class="dt-thumbs">'
                      + "".join(f'<div><img src="{u}"/><span>{lbl}</span></div>' for lbl, u in thumbs) + "</div>",
                      unsafe_allow_html=True)

        def val(name):
            v = p.get(f"pred_{name}") if p is not None else None
            return v if isinstance(v, str) else None

        crumb = " <b>›</b> ".join(esc(x) for x in [row.gender if isinstance(row.gender, str) else "All",
                                                    row.brand if isinstance(row.brand, str) else SOURCE_LABEL.get(row.source),
                                                    str(row.item_type or row.category).title()] if x)
        attrs = [("type", val("article_type")), ("colour", val("colour")), ("pattern", val("pattern")),
                 ("sleeves", val("sleeve_length")), ("neck", val("neck"))]
        rows_html = "".join(f'<div class="k">{k_}</div><div class="v"><span class="sz">{esc(v)}</span></div>'
                            for k_, v in attrs if v)
        about = [x for x in [f"Brand: {row.brand}" if isinstance(row.brand, str) else None,
                             f"Condition: {'pre-owned' if row.is_preowned else 'new'}",
                             f"Material: {row.material}" if isinstance(row.material, str) else None,
                             f"Source: {SOURCE_LABEL.get(row.source)} ({row.year})", f"License: {row.license}"] if x]
        if price_text(row):
            price = price_text(row)
        elif a is not None:
            price = f"{engine.similarities(a['vec'], np.array([engine._row(item_id)]))[0]:.2f} match"
        else:
            price = SOURCE_LABEL.get(row.source, "")
        query = " ".join(str(x) for x in [row.brand, row.title] if isinstance(x, str))
        with right:
            st.markdown(f'<div class="dt-panel"><div class="dt-crumb">{crumb}</div>'
                        f'<div class="dt-title">{esc(row.title)}</div><div class="dt-price">{esc(price)}</div>'
                        f'<div class="dt-grid">{rows_html}<div class="k">more about</div><div class="v"><ul>'
                        + "".join(f"<li>{esc(x)}</li>" for x in about) + "</ul></div></div></div>"
                        f'<a class="dt-cta" href="https://www.google.com/search?tbm=shop&q={quote_plus(query)}" '
                        'target="_blank">find it online ↗</a>', unsafe_allow_html=True)
            if a is not None:
                with st.expander("Why this matches", expanded=True):
                    from silhouette_vision.explain import compare_attributes, occlusion_map, overlay

                    if p is not None:
                        st.markdown("".join(
                            f'<span class="sv-chip {c["relation"]}">{RELATION_MARK[c["relation"]]} {esc(c["label"])}: '
                            f'{esc(c["photo"])}{"" if c["relation"] == "same" else " vs " + esc(c["item"])}</span>'
                            for c in compare_attributes(a["attrs"], p)), unsafe_allow_html=True)
                    if not heat and st.button("Show where the match comes from (a few seconds)", key="heatb"):
                        with st.spinner("Hiding one region at a time and re-measuring the similarity…"):
                            item_img = load_rgb(DATA_ROOT / row.image_path)
                            h1 = overlay(a["query_img"], occlusion_map(engine.encoder, a["query_img"],
                                                                       engine.item_vector(item_id), grid=8, batch=8))
                            h2 = overlay(item_img, occlusion_map(engine.encoder, item_img, a["vec"], grid=8, batch=8))
                            ss.setdefault("heat", {})[item_id] = (pil_uri(h1, 400), pil_uri(h2, 400))
                        release_memory()
                        st.rerun()
                    st.caption("Heatmaps appear in the thumbnail strip: brighter = hiding that region lowers the similarity most.")
            if st.button("‹ Back to results", key="dback"):
                ss.detail = None
                st.rerun()
    with st.container(key="results"):
        st.markdown('<div class="sv-display">More <em>like this</em></div>', unsafe_allow_html=True)
        colour = engine.item_colour(item_id) if match_colour else None
        grid(engine.search(engine.item_vector(item_id), k, filters, exclude=[item_id], query_colour=colour), "sim")


# --- other pages ----------------------------------------------------------------------------------
def page_hero(title: str, sub: str) -> None:
    with st.container(key="herosm"):
        nav_bar()
        st.markdown(f'<div class="sv-hero-title">{title}</div><div class="sv-hero-sub">{sub}</div>', unsafe_allow_html=True)


STYLE_COLOURS = ["#D8F36A", "#E4B6A1", "#F1EEE7", "#9A9A4A", "#C8916F", "#8FB3C9", "#B7A6D9", "#E8C63A",
                 "#6FA37A", "#D87C6B", "#A39E95", "#5E8C8A", "#E59BB3", "#7E7AB8", "#C9C9C9", "#B98A5E"]


def styles_page(engine) -> None:
    page_hero("Style <em>map</em>", "Visual styles found from images alone, and which ones people give away "
              "second-hand versus buy new.")
    clusters, report = get_style_map()
    with st.container(key="section"):
        if clusters is None:
            st.info("Run scripts/05_style_clusters.py to build the style map.")
            return
        category = st.columns([1, 3])[0].selectbox("Category", list(report), format_func=str.capitalize)
        data, info = clusters[clusters.category == category], report[category]
        compare = info.get("donated_share") is not None
        summary = pd.DataFrame([{"Style": v["name"], "Items": v["size"],
                                 **({"Donated share": v["donated_share"]} if compare else {}),
                                 "Signature brands": ", ".join(v["signature_brands"])} for v in info["detail"].values()])
        summary = summary.sort_values("Donated share" if compare else "Items", ascending=False)
        sample = data.sample(min(6000, len(data)), random_state=0).merge(
            engine.catalog[["item_id", "brand", "title"]], on="item_id")
        fig = px.scatter(sample, x="map_x", y="map_y", color="cluster_name", opacity=.78, height=560,
                         hover_data={"brand": True, "title": True, "map_x": False, "map_y": False},
                         labels={"cluster_name": "Style"}, template="plotly_dark", color_discrete_sequence=STYLE_COLOURS)
        fig.update_traces(marker={"size": 4})
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E1D1B",
                          legend_title_text="Style", margin={"l": 0, "r": 0, "t": 10, "b": 0}, font_family="Inter")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summary, hide_index=True, width="stretch", column_config={
            "Donated share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        if compare:
            st.markdown(f'<div class="sv-note">Donated share: second-hand garments given away in 2022–24 vs products sold '
                        f'new in 2025–26 (category average {info["donated_share"]:.0%}). It mixes time with market: new '
                        'products skew premium, donations are Nordic mass-market.</div>', unsafe_allow_html=True)
        style = st.columns([1, 2])[0].selectbox("Show products from a style", summary.Style.tolist())
        members = data[data.cluster_name == style].sample(min(8, (data.cluster_name == style).sum()), random_state=1)
        rows = engine.catalog.set_index("item_id").loc[members.item_id].reset_index().assign(similarity=np.nan)
        grid(rows, "sty", show_similarity=False)


CLOTHING = {"top", "bottoms", "dress", "outerwear"}


def visuelle_image(image_path: str) -> str:
    thumb = DATA_ROOT / "data/processed/thumbs/visuelle" / Path(image_path).with_suffix(".jpg")
    return str(thumb if thumb.exists() else DATA_ROOT / "data/raw/visuelle2/visuelle2/images" / image_path)


def forecast_page(engine) -> None:
    import datetime

    forecaster = get_forecaster()
    photo = ss.get("fc_photo")
    with st.container(key="heroc" if photo else "hero"):
        nav_bar()
        with st.columns([1, 1.35, 1])[1], st.container(key="glass"):
            glass_header(TITLE + " · forecast", close=False)
            with st.container(key="gbody"):
                if photo is None:
                    new = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"], key=f"fc-up-{ss.get('fc_n', 0)}")
                    if new is not None:
                        ss.fc_photo, ss.fc_n = new.getvalue(), ss.get("fc_n", 0) + 1
                        st.rerun()
                else:
                    st.image(photo, width=170)
                st.markdown('<div class="sv-hello" style="margin:.7rem 0 0">How many units would a new product sell '
                            'in its first 12 weeks?</div>', unsafe_allow_html=True)
            if photo is not None:
                with st.container(key="gact-fcnew"):
                    if st.button("New photo", key="gact-fcnew-b", width="stretch"):
                        ss.pop("fc_photo")
                        st.rerun()
    upload = io.BytesIO(photo) if photo else None
    with st.container(key="section"):
        st.markdown('<div class="sv-display">New product <em>forecast</em></div><div class="sv-note">Learned from 5,355 '
                    'launches of Nuna Lie, an Italian fast-fashion womenswear brand (110 stores, 2017–2019; Visuelle 2.0). '
                    'A new product has no sales history, so the model borrows from past launches that look like it.</div>',
                    unsafe_allow_html=True)
        if forecaster is None or upload is None:
            return
        image = load_rgb(upload)
        vec = engine.image_vector(image)
        predictor, _ = get_attributes()
        if predictor is not None:
            kind = next((x["value"] for x in predictor.predict(vec) if x["attribute"] == "article_type"), None)
            if predictor.type_to_category.get(kind) not in CLOTHING | {None}:
                st.warning(f"This looks like **{kind}**. The brand sells women's clothing only, so there's no sales "
                           "history to forecast from.")
                return
        b = forecaster.b
        tags = forecaster.suggest_tags(vec)
        left, right = st.columns([4, 8], gap="large")
        left.markdown(f'<div class="av-card"><div class="av-img" style="height:420px"><img style="max-height:400px;filter:none" '
                      f'src="{pil_uri(image)}"/></div></div>', unsafe_allow_html=True)
        with right:
            c1, c2, c3 = st.columns(3)
            category = c1.selectbox("Category", b["categories"]["category"], index=b["categories"]["category"].index(tags["category"]))
            colour = c2.selectbox("Colour", b["categories"]["color"], index=b["categories"]["color"].index(tags["color"]))
            fabric = c3.selectbox("Fabric", b["categories"]["fabric"], index=b["categories"]["fabric"].index(tags["fabric"]))
            c1, c2, c3 = st.columns(3)
            n_stores = c1.slider("Stores", 1, len(b["store_order"]), int(np.median(b["latest_n_stores"])),
                                 help="Distribution breadth is the strongest predictor: it carries the planners' own expectations.")
            price_pct = c2.slider("Price level", 0, 100, 50, 5, help="Percentile of the brand's latest prices.") / 100
            launch = c3.date_input("Launch date", datetime.date(2019, 9, 2))
            r = forecaster.forecast(vec, category, colour, fabric, price_pct, n_stores, launch)
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected, 12 weeks", f"{r['total']:,.0f} units")
            m2.metric("Likely range (80%)", f"{r['low']:,.0f}–{r['high']:,.0f}",
                      help="On 1,900 unseen 2019 products, actual sales fell in this range for 80% of them.")
            m3.metric("Per store", f"{r['per_store']:.1f} units")
            weeks = pd.DataFrame({"Week": np.arange(1, 13), "Units": r["weekly"]})
            fig = px.bar(weeks, x="Week", y="Units", height=230, template="plotly_dark", color_discrete_sequence=["#D8F36A"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E1D1B", margin={"l": 0, "r": 0, "t": 10, "b": 0})
            st.plotly_chart(fig, width="stretch")
            if forecaster.nearest_similarity(vec) < b["min_similarity"]:
                st.caption("Your photo looks unlike the brand's flat product shots, so treat the forecast as rough.")
        st.markdown('<span class="sv-paper">borrowed from</span>', unsafe_allow_html=True)
        cols = st.columns(8)
        for col, (_, row) in zip(cols, r["lookalikes"].iterrows(), strict=False):
            col.markdown(f'<div class="av-card"><div class="av-img" style="height:170px"><img style="max-height:156px" '
                         f'src="{data_uri(visuelle_image(row.image_path))}"/></div><div class="av-bottom"><span>'
                         f'{esc(row.category)}</span><span>{row.units_per_store:.1f}/store</span></div></div>',
                         unsafe_allow_html=True)
        st.markdown('<div class="sv-note" style="margin-top:1rem">Gradient boosting on store, launch timing, tags, price, '
                    'distribution breadth and look-alike sales; on unseen 2019 products: 34.6% weekly WAPE vs 43.7% for a '
                    'seasonal average.</div>', unsafe_allow_html=True)


def about_page() -> None:
    page_hero("About <em>the project</em>", "Fashion visual search that tells you how sure it is.")
    with st.container(key="section"):
        facts = [("51,047", "products, 2019–2026"), ("88%", "right when it says 80%+ exact"),
                 ("+18 pts", "text search, hybrid"), ("161", "luxury models by name"), ("34.6%", "forecast error vs 43.7%")]
        for col, (v, label) in zip(st.columns(len(facts)), facts, strict=True):
            col.metric(label, v)
        st.markdown('<span class="sv-paper">how it works</span>', unsafe_allow_html=True)
        st.markdown("""
- **Search by photo**: Marqo-FashionSigLIP embeddings, exact search, colour-aware re-ranking, a garment detector to pick one item.
- **Search by words**: the same embeddings plus keyword matching on brand, title and type.
- **What we see**: attribute heads trained on Myntra plus the catalog's own labels (type 81%, colour 88% on recent photos).
- **Exact match**: a calibrated likelihood from how much the best result stands out.
- **Why it matches**: shared attributes and an occlusion heatmap.
- **Style map**: UMAP + k-means per category. **Forecast**: gradient boosting with look-alike launches (Visuelle 2.0).
""")
        st.markdown('<span class="sv-paper">credits</span>', unsafe_allow_html=True)
        st.markdown(ABOUT)


# --- main ---------------------------------------------------------------------------------------
def main():
    ss.setdefault("view", "home")
    ss.setdefault("detail", None)
    st.markdown(css(), unsafe_allow_html=True)
    engine = get_engine()
    filters, k, match_colour = sidebar_filters(engine)
    if ss.detail:
        detail_page(engine, filters, k, match_colour)
    elif ss.view in ("home", "photo", "words"):
        search_page(engine, filters, k, match_colour)
    elif ss.view == "styles":
        styles_page(engine)
    elif ss.view == "forecast":
        forecast_page(engine)
    else:
        about_page()


main()
