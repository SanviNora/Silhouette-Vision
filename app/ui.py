"""Silhouette Vision — Streamlit visual search (Phase 1).

Run: streamlit run app/ui.py
"""

import streamlit as st

from silhouette_vision.config import ROOT
from silhouette_vision.images import load_rgb
from silhouette_vision.search import Filters, SearchEngine

st.set_page_config(page_title="Silhouette Vision", page_icon="👗", layout="wide")

CURRENCY = {"SGD": "S$", "INR": "₹"}
SOURCE_LABEL = {"myntra": "Myntra", "farfetch": "Farfetch"}


@st.cache_resource(show_spinner="Loading model and index…")
def get_engine() -> SearchEngine:
    return SearchEngine("marqo_fashion_siglip")


def price_text(row) -> str:
    symbol = CURRENCY.get(row.currency, "")
    text = f"{symbol}{row.price:,.0f}"
    if row.on_sale and row.discount_pct and row.discount_pct > 0:
        text += f" · −{row.discount_pct:.0%}"
    return text


def sidebar_filters(engine: SearchEngine) -> tuple[Filters, int]:
    st.sidebar.header("Filters")
    cat = engine.catalog
    sources = st.sidebar.multiselect("Catalog", ["myntra", "farfetch"], format_func=SOURCE_LABEL.get)
    categories = st.sidebar.multiselect("Category", sorted(cat.category.unique()))
    genders = st.sidebar.multiselect("Gender", sorted(cat.gender.dropna().unique()))
    condition = st.sidebar.radio("Condition", ["Any", "New", "Pre-owned"], horizontal=True)
    k = st.sidebar.slider("Results", 4, 48, 12, step=4)
    preowned = {"Any": None, "New": False, "Pre-owned": True}[condition]
    st.sidebar.caption(
        f"{len(cat):,} products · Myntra (MIT) + Farfetch (2019 scrape, prices in SGD). "
        "Model: Marqo-FashionSigLIP."
    )
    return Filters(sources, categories, genders, preowned), k


def show_results(results, key_prefix: str):
    if results.empty:
        st.info("No products match these filters.")
        return
    cols = st.columns(4)
    for i, row in results.iterrows():
        with cols[i % 4]:
            st.image(str(ROOT / row.image_path), use_container_width=True)
            badges = " · ".join(
                b for b in [SOURCE_LABEL[row.source], "Pre-owned" if row.is_preowned else None] if b
            )
            st.markdown(f"**{row.brand or ''}**  \n{row.title}")
            st.caption(f"{price_text(row)}  \n{badges} · {row.category} · similarity {row.score:.2f}")
            if st.button("More like this", key=f"{key_prefix}-{row.item_id}"):
                st.session_state.similar_to = row.item_id
                st.rerun()


def main():
    engine = get_engine()
    filters, k = sidebar_filters(engine)

    st.title("Silhouette Vision")
    st.caption("Search fashion products by photo, by description, or both.")

    if item_id := st.session_state.get("similar_to"):
        row = engine.catalog.set_index("item_id").loc[item_id]
        left, right = st.columns([1, 4])
        left.image(str(ROOT / row.image_path), use_container_width=True)
        right.subheader("More like this")
        right.write(f"**{row.brand or ''}** — {row.title}")
        if right.button("Clear"):
            del st.session_state["similar_to"]
            st.rerun()
        show_results(engine.search(engine.item_vector(item_id), k, filters, exclude=[item_id]), "sim")
        return

    photo_tab, text_tab = st.tabs(["Search by photo", "Search by description"])

    with photo_tab:
        upload = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"])
        refine = st.text_input("Optional: refine with text", placeholder="e.g. in red, leather, cropped")
        weight = st.slider("Text influence", 0.0, 0.8, 0.3, 0.05, disabled=not refine)
        if upload:
            image = load_rgb(upload)
            st.image(image, width=180, caption="Your photo")
            query = engine.image_vector(image)
            if refine:
                query = engine.combine(query, engine.text_vector(refine), weight)
            show_results(engine.search(query, k, filters), "img")

    with text_tab:
        text = st.text_input("Describe what you're looking for",
                             placeholder="e.g. black quilted leather shoulder bag with gold chain")
        if text:
            show_results(engine.search(engine.text_vector(text), k, filters), "txt")


main()
