"""Silhouette Vision — Streamlit app.

Run: streamlit run app/ui.py

Tabs: search by photo (garment picker, attributes, named luxury models, exact-match likelihood,
why it matches), search by words (hybrid embedding + keyword search), style map, and a
cold-start demand forecast for a new product.
"""

import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # hosted: no pip install

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from silhouette_vision.config import DATA_REPO, DATA_ROOT, HOSTED, path
from silhouette_vision.images import load_rgb
from silhouette_vision.search import Filters, SearchEngine

st.set_page_config(page_title="Silhouette Vision", page_icon="👗", layout="wide")

SOURCE_LABEL = {"zooclaw": "ZooClaw", "lookbench": "LookBench", "secondhand": "Second-hand",
                "abo": "Amazon"}
EXAMPLE_QUERIES = ["black leather bag", "white sneakers", "floral midi dress", "levi's jeans",
                   "cream cardigan"]
ABOUT = """
Portfolio project, not affiliated with any brand or retailer.
[Code, methods and evaluation](https://github.com/SanviNora/Silhouette-Vision)

**Products (2019–2026)**
- *ZooClaw-Fashion* (2026), SerendipityOne, CC BY-NC 4.0.
- *Second-Hand Fashion* (2022–24), Nauman et al., RISE, Wargön Innovation and Myrorna,
  CC BY 4.0 (doi:10.5281/zenodo.13788681).
- *LookBench* studio gallery (2025), Apache-2.0.
- Footwear: *Amazon Berkeley Objects* (c. 2019–21), Collins et al., CVPR 2022, CC BY 4.0.

**Demand:** *Visuelle 2.0*, Skenderi et al., CVPR Workshops 2022, CC BY-NC-SA 4.0.

**Models:** Marqo-FashionSigLIP, YOLOS-Fashionpedia; attribute heads trained on Fashion Product
Images (Myntra, MIT) and the catalog sources' own labels.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Cormorant Garamond', serif !important; letter-spacing: .01em; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
.sv-hero { padding: 1.4rem 0 .6rem 0; border-bottom: 1px solid #E7DFD4; margin-bottom: 1.2rem; }
.sv-hero h1 { font-size: 3.1rem; margin: 0; line-height: 1.05; color: #1C1917; }
.sv-hero p { color: #6B625A; font-size: 1.02rem; margin: .45rem 0 .8rem 0; max-width: 760px; }
.sv-chip { display: inline-block; padding: .18rem .62rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
  background: #F2ECE3; color: #4A4039; font-size: .78rem; border: 1px solid #E7DFD4; white-space: nowrap; }
.sv-chip.same { background: #E8F1EA; border-color: #CFE3D4; color: #2F6B4F; }
.sv-chip.similar { background: #FBF3E3; border-color: #F0DFBA; color: #8A6A1F; }
.sv-chip.different { background: #F6E9E7; border-color: #EAD0CB; color: #8B3A3A; }
.sv-chip.dark { background: #1C1917; color: #FAF7F2; border-color: #1C1917; }
.sv-section { font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; font-weight: 600;
  margin: 1.4rem 0 .6rem 0; color: #1C1917; }
.sv-muted { color: #78716C; font-size: .86rem; }
.sv-card { background: #FFFFFF; border: 1px solid #ECE5DB; border-radius: 14px; overflow: hidden;
  box-shadow: 0 1px 2px rgba(28,25,23,.04); margin-bottom: .5rem; }
.sv-card .img { height: 250px; background: #FFFFFF; display: flex; align-items: center; justify-content: center;
  position: relative; border-bottom: 1px solid #F1EBE2; }
.sv-card .img img { max-height: 234px; max-width: 92%; object-fit: contain; }
.sv-badge { position: absolute; top: 10px; left: 10px; background: rgba(250,247,242,.94); color: #4A4039;
  font-size: .7rem; padding: .12rem .5rem; border-radius: 999px; border: 1px solid #E7DFD4; }
.sv-sim { position: absolute; top: 10px; right: 10px; background: #1C1917; color: #FAF7F2; font-size: .7rem;
  padding: .12rem .5rem; border-radius: 999px; }
.sv-card .body { padding: .7rem .85rem .75rem .85rem; }
.sv-brand { font-weight: 600; font-size: .9rem; color: #1C1917; min-height: 1.2rem; }
.sv-title { color: #57504A; font-size: .84rem; line-height: 1.3; height: 2.2rem; overflow: hidden; }
.sv-meta { color: #8C837A; font-size: .75rem; margin-top: .35rem; }
.sv-tags { margin-top: .45rem; height: 1.55rem; overflow: hidden; }
.sv-tags .sv-chip { font-size: .7rem; padding: .08rem .45rem; }
.sv-panel { background: #FFFFFF; border: 1px solid #ECE5DB; border-radius: 16px; padding: 1.1rem 1.2rem; margin-bottom: 1rem; }
.sv-panel h3 { margin: 0 0 .3rem 0; font-size: 1.55rem; }
.sv-eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; color: #8C837A; margin-bottom: .2rem; }
.sv-bar { height: 8px; background: #F2ECE3; border-radius: 999px; overflow: hidden; margin: .5rem 0 .25rem 0; }
.sv-bar > div { height: 100%; border-radius: 999px; }
.sv-empty { border: 1.5px dashed #DCD2C4; border-radius: 16px; padding: 2rem 1.5rem; text-align: center; color: #6B625A; background: #FFFDF9; }
div[data-testid="stFileUploader"] section { background: #FFFFFF; border: 1.5px dashed #DCD2C4; border-radius: 14px; }
.stButton > button { border-radius: 999px; border: 1px solid #E0D6C9; background: #FFFFFF; color: #3A332E;
  font-size: .8rem; padding: .25rem .9rem; }
.stButton > button:hover { border-color: #8B3A3A; color: #8B3A3A; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid #E7DFD4; }
.stTabs [data-baseweb="tab"] { font-weight: 500; padding: .5rem .9rem; }
section[data-testid="stSidebar"] { background: #F5F0E8; border-right: 1px solid #E7DFD4; }
[data-testid="stMetricValue"] { font-family: 'Cormorant Garamond', serif; font-size: 2rem; }
.sv-lik { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .2rem 1rem; align-items: baseline; }
</style>
"""


def release_memory() -> None:
    """Return freed memory to the OS after heavy steps. Linux's allocator keeps it otherwise, and
    usage ratchets towards the free host's ~2.7 GB limit. No-op elsewhere."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# --- cached resources -------------------------------------------------------------------------
@st.cache_resource(show_spinner="First start: downloading the catalog (~1–2 minutes)…")
def get_data() -> None:
    """On a fresh host, fetch the data bundle into DATA_ROOT (no-op when the data is there)."""
    if HOSTED:
        from silhouette_vision.bootstrap import ensure_bundle

        ensure_bundle(DATA_ROOT, DATA_REPO)
        release_memory()


@st.cache_resource(show_spinner="Loading the search model…")
def get_engine() -> SearchEngine:
    get_data()
    return SearchEngine("marqo_fashion_siglip")


@st.cache_resource(show_spinner="Loading the garment detector…")
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


# --- small HTML helpers -------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value)) if isinstance(value, str) else ""


def chips(values, kind: str = "") -> str:
    return "".join(f'<span class="sv-chip {kind}">{esc(v)}</span>' for v in values if v)


def price_text(row) -> str:
    """Only Second-Hand garments carry a price: an estimated resale band in Swedish kronor."""
    band = getattr(row, "price_band", None)
    return f"Resale est. {band} SEK" if isinstance(band, str) and band else ""


def product_card(row, tags: list[str], similarity: float | None = None) -> str:
    badge = SOURCE_LABEL.get(row.source, row.source) + (" · pre-owned" if row.is_preowned else "")
    sim = f'<span class="sv-sim">{similarity:.2f}</span>' if similarity is not None else ""
    meta = " · ".join(t for t in [row.category.capitalize() if isinstance(row.category, str) else "",
                                   price_text(row)] if t)
    return (f'<div class="sv-card"><div class="img"><img src="{data_uri(str(DATA_ROOT / row.image_path))}"/>'
            f'<span class="sv-badge">{esc(badge)}</span>{sim}</div><div class="body">'
            f'<div class="sv-brand">{esc(row.brand) or "&nbsp;"}</div><div class="sv-title">{esc(row.title)}</div>'
            f'<div class="sv-meta">{esc(meta)}</div><div class="sv-tags">{chips(tags)}</div></div></div>')


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="sv-section">{esc(title)}</div>'
                + (f'<div class="sv-muted" style="margin-top:-.4rem;margin-bottom:.6rem">{esc(note)}</div>' if note else ""),
                unsafe_allow_html=True)


# --- sidebar ------------------------------------------------------------------------------------
def sidebar_filters(engine: SearchEngine) -> tuple[Filters, int, bool]:
    cat = engine.catalog
    st.sidebar.markdown("### Filters")
    sources = st.sidebar.multiselect("Catalog", list(SOURCE_LABEL), format_func=SOURCE_LABEL.get,
                                     help="Second-hand garments are 31.9k of the 51k products, so they "
                                          "fill many results; pick sources to narrow it down.")
    categories = st.sidebar.multiselect("Category", sorted(cat.category.unique()), format_func=str.capitalize)
    genders = st.sidebar.multiselect("For", sorted(cat.gender.dropna().unique()))
    condition = st.sidebar.radio("Condition", ["Any", "New", "Pre-owned"], horizontal=True)
    k = st.sidebar.slider("Results", 4, 48, 12, step=4)
    match_colour = st.sidebar.toggle(
        "Match colour", value=engine.has_colour, disabled=not engine.has_colour,
        help="Re-ranks photo results toward the photo's colours: colour agreement 50.8% → 54.8% "
             "on held-out garments, for 1.2 points of type agreement.")
    st.sidebar.caption(f"{len(cat):,} products, 2019–2026 · search model: Marqo-FashionSigLIP")
    with st.sidebar.expander("About & credits"):
        st.markdown(ABOUT)
    preowned = {"Any": None, "New": False, "Pre-owned": True}[condition]
    return Filters(sources, categories, genders, preowned), k, match_colour


# --- results ------------------------------------------------------------------------------------
RELATION_MARK = {"same": "✓", "similar": "≈", "different": "✗"}


def card_tags(row, query_attrs) -> list[str]:
    """Attributes the result shares with the photo (✓/≈), else its own main attributes."""
    _, preds = get_attributes()
    if preds is None or row.item_id not in preds.index:
        return []
    from silhouette_vision.enrich import item_tags
    from silhouette_vision.explain import compare_attributes

    if query_attrs:
        comp = compare_attributes(query_attrs, preds.loc[row.item_id])
        shared = [f"{RELATION_MARK[c['relation']]} {c['item']}" for c in comp if c["relation"] != "different"]
        if shared:
            return shared[:3]
    return item_tags(preds.loc[row.item_id])


def show_results(results: pd.DataFrame, key_prefix: str, query_attrs=None, show_similarity=True) -> None:
    if results.empty:
        st.info("No products match these filters.")
        return
    cols = st.columns(4, gap="medium")
    for i, row in results.iterrows():
        with cols[i % 4]:
            st.markdown(product_card(row, card_tags(row, query_attrs), row.similarity if show_similarity else None),
                        unsafe_allow_html=True)
            if st.button("More like this", key=f"{key_prefix}-{row.item_id}", width="stretch"):
                st.session_state.similar_to = row.item_id
                st.rerun()


# --- photo search panels ------------------------------------------------------------------------
def pick_garment(image, key: str = "search"):
    """Whole photo or one detected item; returns the image to search with."""
    detections = get_detector().items(image)
    if not detections:
        st.image(image, width=260)
        return image
    # Items are shown without type labels: the detector (trained on photos of people) mislabels
    # product shots (a denim tote -> "skirt" at 96%). "What we see" then describes the pick.
    crops = [d.crop(image) for d in detections[:5]]
    options = ["Whole photo"] + [f"Item {i + 1}" for i in range(len(crops))]
    cols = st.columns(max(len(options), 4))
    cols[0].image(image, caption="Whole photo", width="stretch")
    for col, crop, label in zip(cols[1:], crops, options[1:], strict=False):
        col.image(crop, caption=label, width="stretch")
    # One item -> search it; several -> start from the whole photo (auto-picking the top item was
    # worse than no crop on LookBench street photos).
    default = 1 if len(detections) == 1 else 0
    choice = st.radio("Search for", options, index=default, horizontal=True, key=f"garment-{key}")
    return image if choice == "Whole photo" else crops[options.index(choice) - 1]


def show_attributes(query_vec) -> list[dict]:
    predictor, _ = get_attributes()
    if predictor is None:
        return []
    attrs = predictor.predict(query_vec)
    if attrs:
        st.markdown('<div class="sv-eyebrow">What we see</div>'
                    + chips([f"{a['label']}: {a['value']} · {a['confidence']:.0%}" for a in attrs]),
                    unsafe_allow_html=True)
        kind = next((a["value"] for a in attrs if a["attribute"] == "article_type"), None)
        if predictor.type_to_category.get(kind) == "jewellery":
            st.info("This looks like jewellery, which the catalog doesn't carry: results will be the "
                    "closest accessories instead.")
    return attrs


def show_named_model(engine: SearchEngine, image_vec) -> None:
    """Name an iconic luxury model (e.g. LV Pochette Félicie) even when the catalog lacks it."""
    recogniser = get_recogniser()
    if recogniser is None:
        return
    from urllib.parse import quote_plus

    from silhouette_vision.named_models import catalog_listings

    (model, prob), *_ = recogniser.recognise(image_vec)
    if prob < 0.5:
        return
    rows = catalog_listings(engine.catalog, model)
    web = f"https://www.google.com/search?tbm=shop&q={quote_plus(model.label)}"
    stock = (f"{len(rows)} listing{'s' if len(rows) != 1 else ''} of this model in the catalog"
             if len(rows) else "Not in our catalog")
    st.markdown(
        f'<div class="sv-panel"><div class="sv-eyebrow">{"Recognised model" if prob >= 0.8 else "Probably"}</div>'
        f'<h3>{esc(model.label)}</h3>{chips([f"Model confidence {prob:.0%}"], "dark")}'
        f'<div class="sv-muted" style="margin-top:.4rem">{esc(stock)} · '
        f'<a href="{web}" target="_blank">Find it online</a></div></div>', unsafe_allow_html=True)
    if len(rows):
        best = rows[np.argsort(-engine.similarities(image_vec, rows))[:4]]
        cols = st.columns(4)
        for col, (_, row) in zip(cols, engine.catalog.iloc[best].iterrows(), strict=False):
            col.image(str(DATA_ROOT / row.image_path), caption=row.title, width="stretch")


TIER_COLOUR = {"Very likely": "#2F6B4F", "Possibly": "#B7791F", "No confident": "#A8A29E"}


def show_best_match(engine: SearchEngine, query, filters, image=None, query_attrs=None) -> None:
    """The single closest product, with a calibrated exact-match likelihood."""
    matcher = get_match_confidence()
    if matcher is None:
        return
    row, top_scores = engine.best_match(query, filters)
    prob = matcher.probability(top_scores)
    tier = matcher.tier(prob)
    colour = next(c for k, c in TIER_COLOUR.items() if tier.startswith(k))
    left, right = st.columns([2, 3], gap="medium")
    with left:
        st.markdown(product_card(row, card_tags(row, query_attrs)), unsafe_allow_html=True)
    with right:
        st.markdown(
            f'<div class="sv-panel"><div class="sv-eyebrow">Best match</div><h3>{esc(tier)}</h3>'
            f'<div class="sv-bar"><div style="width:{prob:.0%};background:{colour}"></div></div>'
            f'<div class="sv-lik"><b>Exact-match likelihood {prob:.0%}</b>'
            f'<span class="sv-muted">visual similarity {row.similarity:.2f}</span></div>'
            f'<div class="sv-muted" style="margin-top:.55rem">Calibrated on benchmark photos: when this says '
            f'80%+, it was the exact product 88% of the time; for products we don\'t stock it claims 80%+ '
            f'for 0.1% of photos. It rewards a result that stands out from the rest, not raw similarity.</div></div>',
            unsafe_allow_html=True)
        if image is not None:
            show_why(engine, image, query_attrs, row)


def show_why(engine: SearchEngine, image, query_attrs, row) -> None:
    """Attribute comparison and where the match comes from, in both photos."""
    from silhouette_vision.explain import compare_attributes, occlusion_map, overlay

    with st.expander("Why this matches"):
        _, preds = get_attributes()
        if query_attrs and preds is not None:
            comp = compare_attributes(query_attrs, preds.loc[row.item_id])
            if comp:
                st.markdown("".join(
                    f'<span class="sv-chip {c["relation"]}">{RELATION_MARK[c["relation"]]} {esc(c["label"])}: '
                    f'{esc(c["photo"])}{"" if c["relation"] == "same" else " vs " + esc(c["item"])}</span>'
                    for c in comp), unsafe_allow_html=True)
        if not st.toggle("Show where the match comes from (a few seconds)", key="why_heatmap"):
            return
        item_image = load_rgb(DATA_ROOT / row.image_path)
        with st.spinner("Hiding one region at a time and re-measuring the similarity…"):
            query_vec, item_vec = engine.image_vector(image), engine.item_vector(row.item_id)
            a, b = st.columns(2)
            a.image(overlay(image, occlusion_map(engine.encoder, image, item_vec, grid=8, batch=8)),
                    caption="Your photo: regions the match depends on", width="stretch")
            b.image(overlay(item_image, occlusion_map(engine.encoder, item_image, query_vec, grid=8, batch=8)),
                    caption="The match: regions that resemble your photo", width="stretch")
        release_memory()
        st.caption("Brighter = hiding this region lowers the similarity most. ≈ marks a close shade or type.")


# --- tabs ---------------------------------------------------------------------------------------
def photo_tab(engine, filters, k, match_colour) -> None:
    left, right = st.columns([5, 7], gap="large")
    with left:
        upload = st.file_uploader("Upload a fashion photo", type=["jpg", "jpeg", "png", "webp"],
                                  label_visibility="collapsed")
        if not upload:
            st.markdown('<div class="sv-empty"><b>Drop a photo of a garment, bag or shoe</b><br>'
                        '<span class="sv-muted">A product shot, a street photo or a screenshot. We find similar '
                        'products, say whether one is the exact item, name iconic luxury models and '
                        'explain the match.</span></div>', unsafe_allow_html=True)
            return
        image = pick_garment(load_rgb(upload), key="search")
        with st.expander("Refine with words"):
            refine = st.text_input("Refinement", placeholder="e.g. in red, leather, cropped",
                                   label_visibility="collapsed")
            weight = st.slider("Text influence", 0.0, 0.8, 0.3, 0.05, disabled=not refine)
    query = engine.image_vector(image)
    with right:
        query_attrs = show_attributes(query)
        show_named_model(engine, query)
        if refine:
            query = engine.combine(query, engine.text_vector(refine), weight)
        show_best_match(engine, query, filters, image, query_attrs)
    colour = engine.image_colour(image) if match_colour else None
    section("Similar products", "✓ shared and ≈ close attributes with your photo; numbers are visual similarity.")
    show_results(engine.search(query, k, filters, query_colour=colour), "img", query_attrs)


def text_tab(engine, filters, k) -> None:
    if "pending_query" in st.session_state:  # an example was clicked on the previous run
        st.session_state.text_query = st.session_state.pop("pending_query")
    text = st.text_input("Describe what you're looking for", key="text_query",
                         placeholder="e.g. black quilted leather shoulder bag with gold chain")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, example in zip(cols, EXAMPLE_QUERIES, strict=True):
        if col.button(example, key=f"ex-{example}", width="stretch"):
            st.session_state.pending_query = example
            st.rerun()
    if text:
        section(f"Results for “{text}”", "Matched on both look (image embedding) and words (brand, title, type).")
        # No similarity badge: text-image cosines are small by nature (~0.1) and would read as poor.
        show_results(engine.search(engine.text_vector(text), k, filters, keywords=text), "txt",
                     show_similarity=False)


def style_map_tab(engine) -> None:
    clusters, report = get_style_map()
    if clusters is None:
        st.info("Run scripts/05_style_clusters.py to build the style map.")
        return
    st.markdown('<div class="sv-muted">Visual styles found from images alone (UMAP + k-means on Marqo '
                'embeddings, after removing each source\'s photo-setup average), named by the model\'s '
                'most distinctive style words.</div>', unsafe_allow_html=True)
    category = st.columns([1, 2])[0].selectbox("Category", list(report), format_func=str.capitalize)
    data = clusters[clusters.category == category]
    info = report[category]
    compare = info.get("donated_share") is not None
    summary = pd.DataFrame([{"Style": v["name"], "Items": v["size"],
                             **({"Donated share": v["donated_share"]} if compare else {}),
                             "Signature brands": ", ".join(v["signature_brands"])}
                            for v in info["detail"].values()])
    summary = summary.sort_values("Donated share" if compare else "Items", ascending=False)
    sample = data.sample(min(6000, len(data)), random_state=0).merge(
        engine.catalog[["item_id", "brand", "title"]], on="item_id")
    fig = px.scatter(sample, x="map_x", y="map_y", color="cluster_name", opacity=0.65,
                     hover_data={"brand": True, "title": True, "map_x": False, "map_y": False},
                     labels={"cluster_name": "Style"}, height=540,
                     color_discrete_sequence=px.colors.qualitative.Safe + px.colors.qualitative.Pastel)
    fig.update_traces(marker={"size": 4})
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title_text="Style",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "Donated share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    if compare:
        st.caption(f"Donated share: second-hand garments given away in 2022–24 vs products sold new in "
                   f"2025–26 (category average {info['donated_share']:.0%}). Above average = over-represented "
                   "among donations. It mixes time with market: new products skew premium, donations are "
                   "Nordic mass-market.")
    style = st.selectbox("Show products from a style", summary.Style.tolist())
    members = data[data.cluster_name == style].sample(min(8, (data.cluster_name == style).sum()), random_state=1)
    rows = engine.catalog.set_index("item_id").loc[members.item_id].reset_index()
    cols = st.columns(4, gap="medium")
    for i, row in rows.iterrows():
        cols[i % 4].markdown(product_card(row, []), unsafe_allow_html=True)


CLOTHING = {"top", "bottoms", "dress", "outerwear"}


def visuelle_image(image_path: str) -> str:
    """Bundle thumbnail if present, else the original Visuelle PNG."""
    thumb = DATA_ROOT / "data/processed/thumbs/visuelle" / Path(image_path).with_suffix(".jpg")
    return str(thumb if thumb.exists() else DATA_ROOT / "data/raw/visuelle2/visuelle2/images" / image_path)


def forecast_tab(engine) -> None:
    import datetime

    forecaster = get_forecaster()
    if forecaster is None:
        st.info("Run scripts/08_visuelle_prepare.py and scripts/09_demand.py to build the demand model.")
        return
    st.markdown('<div class="sv-muted">How many units would a new product sell in its first 12 weeks? '
                'Learned from 5,355 launches of Nuna Lie, an Italian fast-fashion womenswear brand (110 '
                'stores, 2017–2019; Visuelle 2.0). A new product has no sales history, so the model '
                'borrows from past launches that look like it.</div>', unsafe_allow_html=True)
    left, right = st.columns([5, 7], gap="large")
    with left:
        upload = st.file_uploader("Upload a photo of the new product", type=["jpg", "jpeg", "png", "webp"],
                                  key="forecast-upload")
        if not upload:
            return
        image = pick_garment(load_rgb(upload), key="forecast")
    vec = engine.image_vector(image)
    predictor, _ = get_attributes()
    if predictor is not None:
        kind = next((a["value"] for a in predictor.predict(vec) if a["attribute"] == "article_type"), None)
        if predictor.type_to_category.get(kind) not in CLOTHING | {None}:
            right.warning(f"This looks like **{kind}**. The brand in this dataset sells women's clothing "
                          "only, so there is no sales history to forecast from.")
            return
    b = forecaster.b
    tags = forecaster.suggest_tags(vec)
    with right:
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Category", b["categories"]["category"],
                                index=b["categories"]["category"].index(tags["category"]))
        colour = c2.selectbox("Colour", b["categories"]["color"], index=b["categories"]["color"].index(tags["color"]))
        fabric = c3.selectbox("Fabric", b["categories"]["fabric"],
                              index=b["categories"]["fabric"].index(tags["fabric"]))
        st.caption("Suggested from the most similar past launches; change them if they're wrong.")
        c1, c2, c3 = st.columns(3)
        n_stores = c1.slider("Stores", 1, len(b["store_order"]), int(np.median(b["latest_n_stores"])),
                             help="Distribution breadth is the strongest predictor: it carries the "
                                  "planners' own expectations.")
        price_pct = c2.slider("Price level", 0, 100, 50, 5, help="Percentile of the brand's latest prices.") / 100
        launch = c3.date_input("Launch date", datetime.date(2019, 9, 2))
        r = forecaster.forecast(vec, category, colour, fabric, price_pct, n_stores, launch)
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected, 12 weeks", f"{r['total']:,.0f} units")
        m2.metric("Likely range (80%)", f"{r['low']:,.0f} – {r['high']:,.0f}",
                  help="On 1,900 unseen 2019 products, actual sales fell in this range for 80% of them.")
        m3.metric("Per store", f"{r['per_store']:.1f} units")
        weeks = pd.DataFrame({"Week": np.arange(1, 13), "Units": r["weekly"]})
        fig = px.bar(weeks, x="Week", y="Units", height=230, color_discrete_sequence=["#8B3A3A"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")
        if forecaster.nearest_similarity(vec) < b["min_similarity"]:
            st.caption("Your photo looks unlike the brand's flat product shots (people or backgrounds), "
                       "so treat the forecast as rough.")
    section("Borrowed from", "The most similar past launches and what they sold.")
    cols = st.columns(8)
    for col, (_, row) in zip(cols, r["lookalikes"].iterrows(), strict=False):
        col.image(visuelle_image(row.image_path), width="stretch")
        col.caption(f"{row.category} · {row.color}  \n{row.units_per_store:.1f} units/store · {row.season}")
    st.caption("Gradient boosting on store, launch timing, tags, price, distribution breadth and "
               "look-alike sales; on unseen 2019 products: 34.6% weekly WAPE vs 43.7% for a seasonal average.")


def more_like_this(engine, filters, k, match_colour, item_id) -> None:
    row = engine.catalog.iloc[engine._row(item_id)]
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown(product_card(row, []), unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="sv-eyebrow">More like this</div><h2 style="margin-top:0">'
                    f'{esc(row.brand) or ""} {esc(row.title)}</h2>', unsafe_allow_html=True)
        if st.button("← Back to search"):
            del st.session_state["similar_to"]
            st.rerun()
    colour = engine.item_colour(item_id) if match_colour else None
    section("Similar products")
    show_results(engine.search(engine.item_vector(item_id), k, filters, exclude=[item_id], query_colour=colour), "sim")


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    engine = get_engine()
    filters, k, match_colour = sidebar_filters(engine)
    counts = engine.catalog.source.value_counts()
    st.markdown(
        '<div class="sv-hero"><h1>Silhouette Vision</h1><p>Search fashion by photo or by words, see whether '
        'a result is the exact item, and understand why it matches.</p>'
        + chips([f"{len(engine.catalog):,} products", "2019–2026"]
                + [f"{SOURCE_LABEL[s]} {n:,}" for s, n in counts.items()]) + "</div>",
        unsafe_allow_html=True)
    if item_id := st.session_state.get("similar_to"):
        more_like_this(engine, filters, k, match_colour, item_id)
        return
    photo, words, styles, forecast = st.tabs(["Search by photo", "Search by words", "Style map",
                                              "New product forecast"])
    with photo:
        photo_tab(engine, filters, k, match_colour)
    with words:
        text_tab(engine, filters, k)
    with styles:
        style_map_tab(engine)
    with forecast:
        forecast_tab(engine)


main()
